import json
import numpy as np
import torch
from .paths import DatasetPaths
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates


class LesionSequenceDataset(Dataset):
    """
    Dataset for the ConvLSTM baseline (LesionLSTM). Unlike the INR dataset,
    which samples a sparse point cloud from a single random timepoint, this
    dataset returns whole-volume occupancy *grids* for a sequence of visits,
    because the recurrent model consumes and predicts full spatial state at
    each step (frame t -> frame t+1 must be spatially aligned).

    __getitem__ returns:
        grids:        [T, 1, D, H, W] resampled occupancy grids, chronological
        times:        [T] absolute day numbers for each grid
        embedding_id: same convention as the INR (patient_idx * 100 + label_id)

    T is a random contiguous window (length <= max_sequence_len) of the
    trajectory's training-eligible dates, re-sampled every call for
    per-epoch stochasticity (mirrors the INR's random single-timepoint pick,
    just windowed instead of pointwise).

    All timepoints of a trajectory are resampled onto a *shared* canonical
    grid: same physical center + extent, computed once from the union of
    lesion voxels across every visit (plus a margin). This is required for
    the recurrence to be meaningful -- without it, voxel (i,j,k) wouldn't
    refer to the same anatomical location across the sequence.

    Because that canonical-grid computation requires loading every date once,
    and resampling isn't free, resampled grids are cached per trajectory
    index after first access. This is safe: the grids themselves are
    deterministic given a fixed grid_size/margin, only *which window* is
    sampled changes per epoch.

    Note on class imbalance: the INR handled this via positive/border/
    background point sampling. That doesn't apply here since we supervise
    every voxel in the grid -- use `pos_weight` in BCEWithLogitsLoss instead.
    """

    def __init__(self,
                 trajectories,
                 grid_size=(64, 64, 64),
                 margin_mm=20.0,
                 min_extent_mm=80.0,
                 max_sequence_len=6,
                 device='cuda'):
        self.device = device
        self.trajectories = trajectories
        self.grid_size = grid_size
        self.margin_mm = margin_mm
        self.min_extent_mm = min_extent_mm
        self.max_sequence_len = max_sequence_len

        with open(DatasetPaths().patient_to_idx(), "r") as f:
            self.patient_to_idx = json.load(f)

        with open(DatasetPaths().validation_samples(), "r") as f:
            self.validation_samples = json.load(f)
            self.validation_patients = [
                sample["Patient"] + "_" + sample["Lesion"] for sample in self.validation_samples
            ]

        # idx -> {"dates": set, "grids": {date: array}, "times": {date: float}}
        self._sequence_cache = {}

    def __len__(self):
        return len(self.trajectories)

    def get_embedding_id(self, trj):
        return np.int64((self.patient_to_idx[trj.patient_id] * 100) + trj.label_id)

    def _train_dates(self, trj):
        """Same train/val exclusion logic as the INR dataset: for validation
        patients, hold out the last date (extrapolation target) and the
        middle date (interpolation target) from the training pool."""
        dates_list = list(trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates)
        if f"{trj.patient_id}_{trj.label_id}" in self.validation_patients:
            dates_list = dates_list[:-1]
            half_of_the_list = len(dates_list) // 2
            del dates_list[half_of_the_list]
        return dates_list

    def _resample_to_grid(self, labels, affine, center_phys, extent_mm):
        """Nearest-neighbour resample a binary label volume onto a fixed-size
        cubic grid of side `extent_mm`, centered at `center_phys` (physical
        mm coordinates)."""
        d, h, w = self.grid_size
        lin_d = np.linspace(-extent_mm / 2, extent_mm / 2, d)
        lin_h = np.linspace(-extent_mm / 2, extent_mm / 2, h)
        lin_w = np.linspace(-extent_mm / 2, extent_mm / 2, w)
        gd, gh, gw = np.meshgrid(lin_d, lin_h, lin_w, indexing='ij')
        phys_coords = np.stack([gd.ravel(), gh.ravel(), gw.ravel()], axis=-1) + center_phys

        inv_affine = np.linalg.inv(affine)
        ones = np.ones((phys_coords.shape[0], 1))
        homo = np.concatenate([phys_coords, ones], axis=1)
        voxel_coords = (inv_affine @ homo.T).T[:, :3]

        sampled = map_coordinates(
            labels.astype(np.float32), voxel_coords.T, order=0, mode='constant', cval=0.0
        )
        return sampled.reshape(d, h, w).astype(np.float32)

    def _load_and_resample_trajectory(self, idx, dates_needed):
        """Load every requested date once, compute (or reuse) a shared
        canonical crop for this trajectory, resample all requested dates onto
        it. Cached per trajectory index; grows the cache if new dates are
        requested later (e.g. validation subclasses asking for a held-out
        date not in the training pool)."""
        cached = self._sequence_cache.get(idx)
        missing = [d for d in dates_needed if cached is None or d not in cached["grids"]]

        if not missing:
            return cached

        trj = self.trajectories[idx]
        loaded = {}
        all_pos_phys = []
        for date in dates_needed:
            (labels, time_point), affine = trj.load_labels_for_inr(
                selected_date=date, absolute_day_number=True, affine=True
            )
            loaded[date] = (labels, affine, time_point)
            pos_vox = np.argwhere(labels)
            if len(pos_vox) > 0:
                ones = np.ones((len(pos_vox), 1))
                homo = np.concatenate([pos_vox, ones], axis=1)
                pos_phys = (affine @ homo.T).T[:, :3]
                all_pos_phys.append(pos_phys)

        if len(all_pos_phys) > 0:
            all_pos_phys = np.concatenate(all_pos_phys, axis=0)
            bbox_min = all_pos_phys.min(axis=0)
            bbox_max = all_pos_phys.max(axis=0)
            center_phys = (bbox_min + bbox_max) / 2.0
            extent_mm = float(np.maximum(
                (bbox_max - bbox_min + 2 * self.margin_mm).max(), self.min_extent_mm
            ))
        else:
            # No lesion voxels in any requested date: fall back to the first
            # date's whole-volume center, matching the INR's "no positive
            # coords" fallback.
            first_labels, first_affine, _ = loaded[dates_needed[0]]
            volume_center_voxel = (np.array(first_labels.shape) - 1) / 2.0
            center_phys = first_affine[:3, :3] @ volume_center_voxel + first_affine[:3, 3]
            extent_mm = self.min_extent_mm

        grids = dict(cached["grids"]) if cached else {}
        times = dict(cached["times"]) if cached else {}
        for date in dates_needed:
            labels, affine, time_point = loaded[date]
            grids[date] = self._resample_to_grid(labels, affine, center_phys, extent_mm)
            times[date] = time_point

        cached = {"dates": set(grids.keys()), "grids": grids, "times": times,
                  "center": center_phys, "extent": extent_mm}
        self._sequence_cache[idx] = cached
        return cached

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        dates_list = self._train_dates(trj)

        if len(dates_list) < 2:
            # Not enough history to form even one transition. Degenerate
            # fallback: duplicate the single date so the model still gets a
            # valid (if uninformative) training pair.
            dates_list = dates_list * 2

        max_len = min(len(dates_list), self.max_sequence_len)
        if len(dates_list) > max_len:
            start = np.random.randint(0, len(dates_list) - max_len + 1)
            window_dates = dates_list[start:start + max_len]
        else:
            window_dates = dates_list

        cached = self._load_and_resample_trajectory(idx, dates_list)

        grids = np.stack([cached["grids"][d] for d in window_dates], axis=0)[:, None]  # [T,1,D,H,W]
        times = np.array([cached["times"][d] for d in window_dates], dtype=np.float32)

        return grids, times, self.get_embedding_id(trj)


class Validation_Extrapolation_LesionSequenceDataset(LesionSequenceDataset):
    """
    Returns the full training history plus the held-out final visit as an
    explicit target, so the caller can run teacher-forced history through the
    model and score exactly one forecasted step against a real future visit:

        preds = model(history_grids, torch.cat([history_times, target_time[:,None]], 1),
                       patient_idx, teacher_forcing=False, n_future=1)
        loss  = bce(preds[:, -1], target_grid)
    """
    def __init__(self, trajectories, **kwargs):
        super().__init__(trajectories, **kwargs)
        self.trajectories = [
            trj for trj in self.trajectories
            if f"{trj.patient_id}_{trj.label_id}" in self.validation_patients
        ]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        full_dates = list(trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates)
        history_dates = self._train_dates(trj)
        target_date = full_dates[-1]  # held-out extrapolation target

        cached = self._load_and_resample_trajectory(idx, history_dates + [target_date])

        history_grids = np.stack([cached["grids"][d] for d in history_dates], axis=0)[:, None]
        history_times = np.array([cached["times"][d] for d in history_dates], dtype=np.float32)
        target_grid = cached["grids"][target_date][None]
        target_time = np.float32(cached["times"][target_date])

        return history_grids, history_times, target_grid, target_time, self.get_embedding_id(trj)


class Validation_Interpolation_LesionSequenceDataset(LesionSequenceDataset):
    """
    Returns the visits strictly before the held-out interpolation date as
    history, and that date's grid as the target. See the module-level note
    at the top of this file: this is a *forecast from the past* toward an
    intermediate timestamp, not a true bidirectional interpolation -- the
    model never sees the later, still-observed visits the INR implicitly
    benefits from through its jointly-fit per-patient latent. Worth flagging
    explicitly if this baseline is compared against the INR's interpolation
    numbers.
    """
    def __init__(self, trajectories, **kwargs):
        super().__init__(trajectories, **kwargs)
        self.trajectories = [
            trj for trj in self.trajectories
            if f"{trj.patient_id}_{trj.label_id}" in self.validation_patients
        ]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        full_dates = list(trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates)

        # NOTE: the original Validation_Interpolation_LesionDataset computed
        # `half_of_the_list = len(dates_list[-1]) // 2`, where dates_list[-1]
        # is a single date, not a list -- that looks like a typo for
        # `len(dates_list) // 2`. Using the corrected version here; flag if
        # the original was intentional for a reason not visible from this
        # file alone.
        dates_minus_last = full_dates[:-1]
        half_of_the_list = len(dates_minus_last) // 2
        target_date = dates_minus_last[half_of_the_list]

        target_pos = full_dates.index(target_date)
        history_dates = full_dates[:target_pos]
        if len(history_dates) == 0:
            # target is the first visit: no history available, degenerate case
            history_dates = [target_date]

        cached = self._load_and_resample_trajectory(idx, history_dates + [target_date])

        history_grids = np.stack([cached["grids"][d] for d in history_dates], axis=0)[:, None]
        history_times = np.array([cached["times"][d] for d in history_dates], dtype=np.float32)
        target_grid = cached["grids"][target_date][None]
        target_time = np.float32(cached["times"][target_date])

        return history_grids, history_times, target_grid, target_time, self.get_embedding_id(trj)


class Plotting_LesionSequenceDataset(LesionSequenceDataset):
    """Returns the full, chronologically-ordered sequence for a trajectory
    (all training-eligible dates), attached to the trajectory object -- for
    qualitative rollout visualizations, mirroring Plotting_LesionDataset."""

    def __init__(self, trajectories, **kwargs):
        super().__init__(trajectories, **kwargs)
        with open(DatasetPaths().plotting_samples(), "r") as f:
            self.plotting_samples = [
                sample["Patient"] + "_" + sample["Lesion"] for sample in json.load(f)
            ]
        self.trajectories = [
            trj for trj in self.trajectories
            if f"{trj.patient_id}_{trj.label_id}" in self.plotting_samples
        ]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        dates_list = list(trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates)

        cached = self._load_and_resample_trajectory(idx, dates_list)

        grids = np.stack([cached["grids"][d] for d in dates_list], axis=0)[:, None]
        times = np.array([cached["times"][d] for d in dates_list], dtype=np.float32)

        trj.embedding_id = self.get_embedding_id(trj)
        trj.grids = grids
        trj.times = times
        trj.canonical_center = cached["center"]
        trj.canonical_extent_mm = cached["extent"]

        return trj


def sequence_collate_fn(batch):
    """
    Batching collate_fn for LesionSequenceDataset. Sequence lengths T vary
    across trajectories, so this pads to the max T in the batch with the
    first frame's value (safe no-op padding: LesionLSTM.forward only ever
    reads times[:, step+1] as the conditioning target and grids up to the
    real length) and also returns a `lengths` tensor.

    Recommended default, though, is simply `batch_size=1` in the DataLoader --
    3D ConvLSTM grids are already memory-heavy per sample, so padding rarely
    buys much, and it avoids needing to mask the loss to the valid length
    (LesionLSTM.forward as given does not do that masking itself).
    """
    grids_list, times_list, ids_list = zip(*batch)
    lengths = torch.tensor([g.shape[0] for g in grids_list], dtype=torch.long)
    T_max = int(lengths.max())

    B = len(batch)
    _, C, D, H, W = grids_list[0].shape
    padded_grids = torch.zeros(B, T_max, C, D, H, W, dtype=torch.float32)
    padded_times = torch.zeros(B, T_max, dtype=torch.float32)

    for b, (g, t) in enumerate(zip(grids_list, times_list)):
        T = g.shape[0]
        padded_grids[b, :T] = torch.from_numpy(g)
        padded_times[b, :T] = torch.from_numpy(t)
        if T < T_max:
            padded_grids[b, T:] = padded_grids[b, T - 1]
            padded_times[b, T:] = padded_times[b, T - 1]

    patient_idx = torch.tensor(ids_list, dtype=torch.long)
    return padded_grids, padded_times, patient_idx, lengths