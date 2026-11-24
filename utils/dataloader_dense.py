import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from .dataloader import LesionDataset


@dataclass
class _LesionCrop:
    """All training scans of one lesion, cropped to the union of their masks.

    Every positive voxel of every training scan lies inside the crop, so any
    voxel outside it is background for every scan. The margin is large enough
    for the border band (binary dilation), so cropping changes no label and no
    sampling probability; it only avoids full-volume work per item.
    """
    shape: tuple                 # full volume shape
    affine: np.ndarray
    origin: np.ndarray           # [3] voxel index of the crop corner
    masks: dict                  # date -> bool crop
    counts: np.ndarray           # number of training scans positive per crop voxel
    times: dict                  # date -> time point


def _crop_lookup(crop, origin, voxels):
    """Values of ``crop`` at full-volume ``voxels``; False/0 outside the crop."""
    local = voxels - origin
    inside = np.all((local >= 0) & (local < np.asarray(crop.shape)), axis=1)
    values = np.zeros(len(voxels), dtype=crop.dtype)
    idx = local[inside]
    values[inside] = crop[idx[:, 0], idx[:, 1], idx[:, 2]]
    return values


_PRELOAD_DATASET = None


def _preload_worker(job):
    key, trj, dates = job
    return key, _PRELOAD_DATASET._build_crop(trj, dates)


class DenseLesionDataset(LesionDataset):
    """Sample observed labels and temporal constraints from dense lesion regions.

    Each item holds out one training scan from the core/union calculation,
    supervises its real mask at its actual time, and adds intermediate-time
    samples constrained by the other scans' intersection and union.

    Every lesion is loaded once (``preload``, in parallel, before the
    DataLoader forks) and kept as a small crop around the union of its
    training masks. Background samples far from the lesion are drawn by
    uniform rejection sampling over the full volume, which gives the same
    distribution as sampling from the list of all background voxels.
    """

    def __init__(
        self,
        trajectories,
        device="cuda",
        dialation_iterations=5,
        background_samples=3000,
        oversample_positive_samples=True,
        preload_processes=8,
        near_field_fraction=0.0,
        near_field_margin_mm=20.0,
        voxel_jitter=False,
        single_time_items=False,
    ):
        if background_samples < 3:
            raise ValueError("background_samples must be at least 3")

        super().__init__(
            trajectories,
            device=device,
            dialation_iterations=dialation_iterations,
            background_samples=background_samples,
            oversample_positive_samples=oversample_positive_samples,
        )
        if not 0.0 <= near_field_fraction <= 1.0:
            raise ValueError("near_field_fraction must be in [0, 1]")
        # Share of the far-background negatives that is instead drawn from the
        # crop box grown by near_field_margin_mm (0 = previous behaviour).
        self.near_field_fraction = float(near_field_fraction)
        self.near_field_margin_mm = float(near_field_margin_mm)
        # Move each sample uniformly inside its voxel (+-0.5 voxel per axis) instead
        # of the voxel centre. With 5-6.5 mm slices the centres all lie on the
        # slice planes and the space between slices is never supervised; the
        # label stays the voxel's, i.e. nearest-voxel like the grid validation.
        self.voxel_jitter = bool(voxel_jitter)
        # The models read ONE time per item (INR: time of point 0, NODE: mean
        # time). Mixed items (2/3 of the points at the target scan time, 1/3 at
        # random intermediate times) therefore train with wrong times. With this
        # flag an item is either a scan item (all points at the scan time) or,
        # with probability 1/3, an intermediate item (all points at one random
        # time), which keeps the same overall share of each kind of point.
        self.single_time_items = bool(single_time_items)
        self._crops = {}
        if preload_processes:
            self.preload(processes=preload_processes)

    def _training_dates(self, trj):
        dates = list(trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates)
        lesion_id = f"{trj.patient_id}_{trj.label_id}"
        if lesion_id in self.validation_samples:
            validation = self.validation_samples[lesion_id]
            extrapolation_dates = validation["extrapolation_dates"]
            dates = [date for date in dates if date not in extrapolation_dates]
            interpolation_index = int(validation["interpolation"])
            if interpolation_index >= 0:  # -1 = no interpolation scan
                dates.pop(interpolation_index)
        return dates

    def _load_scan(self, trj, date):
        (labels, time_point), affine = trj.load_labels_for_inr(
            selected_date=date,
            absolute_day_number=True,
            affine=True,
        )
        return np.asarray(labels, dtype=np.uint8), float(time_point), affine

    # ------------------------------------------------------------------
    # Per-lesion crop
    # ------------------------------------------------------------------
    def _build_crop(self, trj, dates):
        scans = {}
        affine = shape = None
        for date in dates:
            labels, time_point, scan_affine = self._load_scan(trj, date)
            if shape is None:
                shape, affine = labels.shape, scan_affine
            elif labels.shape != shape:
                raise ValueError(
                    f"Label shape changed for lesion {trj.patient_id}_{trj.label_id}"
                )
            elif not np.allclose(scan_affine, affine):
                raise ValueError(
                    f"Affine changed for lesion {trj.patient_id}_{trj.label_id}"
                )
            scans[date] = (labels > 0, time_point)

        union = np.zeros(shape, dtype=bool)
        for mask, _ in scans.values():
            union |= mask
        positive = np.argwhere(union)
        if len(positive):
            # The border band reaches dialation_iterations voxels from a lesion.
            margin = int(self.dialation_iterations) + 1
            lo = np.maximum(positive.min(axis=0) - margin, 0)
            hi = np.minimum(positive.max(axis=0) + margin + 1, np.asarray(shape))
        else:
            lo = np.zeros(3, dtype=np.int64)
            hi = np.ones(3, dtype=np.int64)
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))

        masks = {date: mask[crop].copy() for date, (mask, _) in scans.items()}
        count_dtype = np.uint16 if len(dates) <= np.iinfo(np.uint16).max else np.uint32
        counts = np.zeros(masks[dates[0]].shape, dtype=count_dtype)
        for mask in masks.values():
            counts += mask
        return _LesionCrop(
            shape=tuple(int(s) for s in shape),
            affine=np.asarray(affine, dtype=np.float64),
            origin=lo.astype(np.int64),
            masks=masks,
            counts=counts,
            times={date: time_point for date, (_, time_point) in scans.items()},
        )

    @staticmethod
    def _crop_key(trj, dates):
        return (trj.patient_id, trj.label_id, tuple(dates))

    def _crop(self, trj, dates):
        key = self._crop_key(trj, dates)
        crop = self._crops.get(key)
        if crop is None:
            crop = self._build_crop(trj, dates)
            self._crops[key] = crop
        return crop

    def preload(self, processes=8):
        """Build every lesion's crop in parallel (before the DataLoader forks)."""
        global _PRELOAD_DATASET
        jobs = {}
        for trj in self.trajectories:
            dates = self._training_dates(trj)
            key = self._crop_key(trj, dates)
            if len(dates) >= 2 and key not in self._crops:
                jobs[key] = (key, trj, dates)
        if not jobs:
            return
        _PRELOAD_DATASET = self
        try:
            if processes > 1:
                with mp.get_context("fork").Pool(processes) as pool:
                    results = pool.imap_unordered(_preload_worker, jobs.values(), chunksize=2)
                    for key, crop in tqdm(results, total=len(jobs), desc="Preloading lesions"):
                        self._crops[key] = crop
            else:
                for job in tqdm(jobs.values(), desc="Preloading lesions"):
                    key, crop = _preload_worker(job)
                    self._crops[key] = crop
        finally:
            _PRELOAD_DATASET = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_voxels(voxels, count, replace=True):
        if count == 0:
            return np.empty((0, 3), dtype=np.int64)
        if len(voxels) == 0:
            return np.empty((0, 3), dtype=np.int64)
        indices = np.random.choice(len(voxels), size=count, replace=replace or count > len(voxels))
        return voxels[indices]

    @staticmethod
    def _sample_outside(shape, crop_mask, origin, count, lo=None, hi=None):
        """``count`` voxels uniformly from the box [lo, hi) where ``crop_mask`` is False.

        The box defaults to the full volume. Equivalent to drawing with
        replacement from ``np.argwhere(~mask)`` restricted to the box, but
        without listing the millions of background voxels.
        """
        if count == 0:
            return np.empty((0, 3), dtype=np.int64)
        lo = np.zeros(3, dtype=np.int64) if lo is None else np.asarray(lo, dtype=np.int64)
        hi = np.asarray(shape, dtype=np.int64) if hi is None else np.asarray(hi, dtype=np.int64)
        accepted, total = [], 0
        while total < count:
            draw = max(2 * (count - total), 64)
            candidates = np.random.randint(lo, hi, size=(draw, 3))
            keep = candidates[~_crop_lookup(crop_mask, origin, candidates)]
            accepted.append(keep)
            total += len(keep)
        return np.concatenate(accepted)[:count]

    def _near_field_box(self, crop):
        """Crop box (lesion union + border margin) grown by near_field_margin_mm."""
        spacing = np.linalg.norm(crop.affine[:3, :3], axis=0)
        grow = np.ceil(self.near_field_margin_mm / np.maximum(spacing, 1e-6)).astype(np.int64)
        crop_shape = np.asarray(crop.counts.shape, dtype=np.int64)
        lo = np.maximum(crop.origin - grow, 0)
        hi = np.minimum(crop.origin + crop_shape + grow, np.asarray(crop.shape, dtype=np.int64))
        return lo, hi

    def _sample_negatives(self, crop, mask, count):
        """Background negatives; ``near_field_fraction`` of them from the near-field box."""
        near_count = int(round(count * self.near_field_fraction))
        parts = [self._sample_outside(crop.shape, mask, crop.origin, count - near_count)]
        if near_count:
            lo, hi = self._near_field_box(crop)
            parts.append(self._sample_outside(crop.shape, mask, crop.origin, near_count, lo, hi))
        return np.concatenate(parts)

    def _sample_observed(self, crop, target_mask, sample_count):
        positive_voxels = np.argwhere(target_mask) + crop.origin
        negative_total = int(np.prod(crop.shape)) - len(positive_voxels)
        target_positive_count = sample_count // 2

        if len(positive_voxels) == 0:
            positive_count = 0
        elif negative_total == 0:
            positive_count = sample_count
        elif self.oversample_positive_samples:
            positive_count = target_positive_count
        else:
            positive_count = min(target_positive_count, len(positive_voxels))

        negative_count = sample_count - positive_count
        positive_samples = self._sample_voxels(
            positive_voxels,
            positive_count,
            replace=self.oversample_positive_samples,
        )

        # The crop margin covers the full dilation, so this equals the
        # full-volume border band.
        border = binary_dilation(target_mask, iterations=self.dialation_iterations) & ~target_mask
        border_voxels = np.argwhere(border) + crop.origin
        border_count = min(negative_count // 5, len(border_voxels))
        border_samples = self._sample_voxels(
            border_voxels, border_count, replace=False
        )

        remaining_negative_count = negative_count - border_count
        negative_samples = self._sample_negatives(
            crop, target_mask, remaining_negative_count
        )
        samples = np.concatenate(
            [positive_samples, border_samples, negative_samples], axis=0
        )
        return samples, np.concatenate(
            [
                np.ones(len(positive_samples), dtype=np.float32),
                np.zeros(len(border_samples) + len(negative_samples), dtype=np.float32),
            ]
        )

    def _sample_intermediate(self, crop, core, union, sample_count):
        core_voxels = np.argwhere(core) + crop.origin
        outside_count = int(np.prod(crop.shape)) - int(union.sum())
        if len(core_voxels) == 0 and outside_count == 0:
            raise ValueError(
                "No persistent lesion or outside-union voxels available for "
                "intermediate-time supervision"
            )

        if len(core_voxels) and outside_count:
            positive_count = sample_count // 2
        elif len(core_voxels):
            positive_count = sample_count
        else:
            positive_count = 0

        negative_count = sample_count - positive_count
        positive_samples = self._sample_voxels(core_voxels, positive_count)
        negative_samples = self._sample_negatives(crop, union, negative_count)
        voxels = np.concatenate([positive_samples, negative_samples], axis=0)
        labels = np.concatenate(
            [
                np.ones(len(positive_samples), dtype=np.float32),
                np.zeros(len(negative_samples), dtype=np.float32),
            ]
        )
        return voxels, labels

    def _voxel_coords(self, voxels, affine, volume_shape, times):
        voxels = voxels.astype(np.float64)
        if self.voxel_jitter:
            voxels = voxels + np.random.uniform(-0.5, 0.5, size=voxels.shape)
        voxel_indices = np.concatenate(
            [voxels, np.ones((len(voxels), 1), dtype=voxels.dtype)], axis=1
        )
        physical_coords = (affine @ voxel_indices.T).T[:, :3]
        volume_center_voxel = (np.asarray(volume_shape) - 1) / 2.0
        volume_center_phys = affine[:3, :3] @ volume_center_voxel + affine[:3, 3]
        spatial_coords = (physical_coords - volume_center_phys) / 100.0
        return np.column_stack([spatial_coords, times]).astype(np.float32)

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        dates = self._training_dates(trj)
        if len(dates) < 2:
            raise ValueError(
                f"Dense training requires at least two training scans for "
                f"lesion {trj.patient_id}_{trj.label_id}"
            )

        crop = self._crop(trj, dates)
        target_date = str(np.random.choice(dates))
        target_mask = crop.masks[target_date]
        target_time = crop.times[target_date]

        remaining_scan_count = len(dates) - 1
        remaining_positive_counts = crop.counts - target_mask
        core = remaining_positive_counts == remaining_scan_count
        union = remaining_positive_counts > 0

        if self.single_time_items:
            intermediate_item = np.random.rand() < 1.0 / 3.0
            intermediate_count = self.max_samples if intermediate_item else 0
        else:
            intermediate_count = self.max_samples // 3
        observed_count = self.max_samples - intermediate_count

        observed_voxels, observed_labels = self._sample_observed(
            crop, target_mask, observed_count
        )
        intermediate_voxels, intermediate_labels = self._sample_intermediate(
            crop, core, union, intermediate_count
        )

        min_time = min(crop.times[date] for date in dates)
        max_time = max(crop.times[date] for date in dates)
        if self.single_time_items:
            intermediate_times = np.full(
                intermediate_count, np.random.uniform(min_time, max_time), dtype=np.float32
            )
        else:
            intermediate_times = np.random.uniform(
                min_time, max_time, size=intermediate_count
            ).astype(np.float32)

        coords = np.concatenate(
            [
                self._voxel_coords(
                    observed_voxels,
                    crop.affine,
                    crop.shape,
                    np.full(len(observed_voxels), target_time, dtype=np.float32),
                ),
                self._voxel_coords(
                    intermediate_voxels,
                    crop.affine,
                    crop.shape,
                    intermediate_times,
                ),
            ],
            axis=0,
        )
        labels = np.concatenate([observed_labels, intermediate_labels])
        permutation = np.random.permutation(len(labels))
        return (
            coords[permutation],
            labels[permutation],
            self.get_embedding_id(trj),
        )
