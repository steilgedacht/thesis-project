"""Point-sampled sequence data for the Point-LSTM.

Instead of resampling whole 3D volumes onto a grid (ConvLSTM), every scan is
represented by a small set of sampled points ``(x, y, z, label)`` in the same
normalized coordinate frame as the INR / Neural ODE:

    xyz = (physical_mm - volume_center_mm) / 100

One training item is

    history : up to ``max_history_len`` scans strictly before the target,
              each as ``history_points`` sampled points
    queries : ``query_points`` points at the target scan with their labels
              (and the signed distance to the last history lesion)

Date handling (identical to the INR / Neural ODE datasets):
    * training never uses a lesion's ``extrapolation_dates`` or its
      ``interpolation`` date from ``validation_lesions.json``;
    * evaluation history is every non-held-out scan before the target, so the
      interpolation scan is never part of an extrapolation history and the
      target scan is never encoded.

Each scan is reduced once to a compact ``ScanEntry`` (positive voxels plus a
signed distance crop around the lesion, int8-quantized). ``ScanCache.preload``
builds all entries in parallel before the DataLoader forks, so workers share
them copy-on-write instead of re-reading NIfTI files.
"""

import json
import multiprocessing as mp
import os
import zlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, map_coordinates
from torch.utils.data import Dataset
from tqdm import tqdm

from .paths import DatasetPaths


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------
def lesion_key(trj):
    return f"{trj.patient_id}_{trj.label_id}"


def all_dates(trj):
    return [str(d) for d in (trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates)]


def load_validation_samples():
    with open(DatasetPaths().validation_samples(), encoding="utf-8") as file:
        samples = json.load(file)
    return {f"{s['Patient']}_{s['Lesion']}": s for s in samples}


def interpolation_index(sample):
    """Index of the interpolation scan, or None if the lesion has none (-1)."""
    index = int(sample["interpolation"])
    return index if index >= 0 else None


def training_dates(trj, validation_samples):
    """Dates usable for training; same logic as LesionDataset / DenseLesionDataset."""
    dates = all_dates(trj)
    sample = validation_samples.get(lesion_key(trj))
    if sample is None:
        return dates
    extrapolation = {str(d) for d in sample["extrapolation_dates"]}
    remaining = [d for d in dates if d not in extrapolation]
    index = interpolation_index(sample)
    if index is None:
        return remaining
    interpolation_date = remaining.pop(index)
    if interpolation_date != dates[index]:
        raise ValueError(
            f"{lesion_key(trj)}: interpolation index {index} points to "
            "different dates before/after removing extrapolation dates."
        )
    return remaining


def held_out_dates(trj, validation_samples):
    """Validation dates of a lesion (extrapolation dates + interpolation date)."""
    sample = validation_samples.get(lesion_key(trj))
    if sample is None:
        return set()
    held = {str(d) for d in sample["extrapolation_dates"]}
    index = interpolation_index(sample)
    if index is not None:
        held.add(all_dates(trj)[index])
    return held


def evaluation_history(trj, target_date, validation_samples, max_history_len):
    """Observed scans before ``target_date``, excluding every held-out date."""
    dates = all_dates(trj)
    target_index = dates.index(str(target_date))
    held = held_out_dates(trj, validation_samples)
    history = [d for d in dates[:target_index] if d not in held]
    if not history:
        raise ValueError(f"No observed history before {target_date} for {lesion_key(trj)}")
    return history[-max_history_len:]


# ---------------------------------------------------------------------------
# Per-scan representation
# ---------------------------------------------------------------------------
@dataclass
class ScanEntry:
    time: float
    shape: tuple
    affine: np.ndarray
    inverse_affine: np.ndarray
    center_phys: np.ndarray
    spacing: np.ndarray
    positive_voxels: np.ndarray       # [n, 3] int16
    crop_origin: np.ndarray | None    # [3] voxel index of the crop corner
    sdf_q: np.ndarray | None          # int8 signed distance (mm / sdf_scale), <0 inside
    sdf_scale: float
    sdf_clip_mm: float

    @property
    def volume_mm3(self):
        return float(len(self.positive_voxels) * np.prod(self.spacing))

    @property
    def crop_box(self):
        if self.sdf_q is None:
            return np.zeros(3, dtype=np.int64), np.asarray(self.shape, dtype=np.int64)
        return self.crop_origin, self.crop_origin + np.asarray(self.sdf_q.shape)

    def voxels_to_model(self, voxels):
        voxels = np.asarray(voxels, dtype=np.float64)
        phys = voxels @ self.affine[:3, :3].T + self.affine[:3, 3]
        return ((phys - self.center_phys) / 100.0).astype(np.float32)

    def model_to_voxels(self, xyz):
        phys = np.asarray(xyz, dtype=np.float64)[:, :3] * 100.0 + self.center_phys
        return phys @ self.inverse_affine[:3, :3].T + self.inverse_affine[:3, 3]

    def _crop_lookup(self, voxels_int):
        local = voxels_int - self.crop_origin
        inside = np.all((local >= 0) & (local < np.asarray(self.sdf_q.shape)), axis=1)
        values = np.full(len(voxels_int), self.sdf_clip_mm, dtype=np.float32)
        idx = local[inside]
        values[inside] = self.sdf_q[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32) * self.sdf_scale
        return values

    def sdf_at_voxels(self, voxels_int):
        if self.sdf_q is None:
            return np.full(len(voxels_int), self.sdf_clip_mm, dtype=np.float32)
        return self._crop_lookup(np.asarray(voxels_int, dtype=np.int64))

    def label_at_voxels(self, voxels_int):
        return (self.sdf_at_voxels(voxels_int) < 0).astype(np.float32)

    def sdf_at_model(self, xyz):
        """Trilinear signed distance (mm) at continuous model coordinates."""
        if self.sdf_q is None:
            return np.full(len(xyz), self.sdf_clip_mm, dtype=np.float32)
        local = (self.model_to_voxels(xyz) - self.crop_origin).T
        values = map_coordinates(
            self.sdf_q.astype(np.float32), local, order=1, mode="constant", cval=127.0
        )
        return np.clip(values * self.sdf_scale, -self.sdf_clip_mm, self.sdf_clip_mm).astype(np.float32)

    def ring_voxels(self, ring_mm):
        if self.sdf_q is None:
            return np.empty((0, 3), dtype=np.int64)
        threshold = max(1, int(np.ceil(ring_mm / self.sdf_scale)))
        return np.argwhere(np.abs(self.sdf_q.astype(np.int16)) <= threshold) + self.crop_origin


def build_scan_entry(trj, date, sdf_clip_mm=20.0):
    (labels, time_point), affine = trj.load_labels_for_inr(
        selected_date=str(date), absolute_day_number=True, affine=True
    )
    labels = np.asarray(labels) > 0
    shape = tuple(int(s) for s in labels.shape)
    affine = np.asarray(affine, dtype=np.float64)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    center_voxel = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    center_phys = affine[:3, :3] @ center_voxel + affine[:3, 3]
    positive = np.argwhere(labels)
    sdf_scale = sdf_clip_mm / 127.0

    crop_origin, sdf_q = None, None
    if len(positive):
        margin = np.ceil(sdf_clip_mm / spacing).astype(np.int64) + 1
        lo = np.maximum(positive.min(axis=0) - margin, 0)
        hi = np.minimum(positive.max(axis=0) + margin + 1, np.asarray(shape))
        crop = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        half_voxel = 0.5 * float(spacing.min())
        dist_out = distance_transform_edt(~crop, sampling=spacing)
        dist_in = distance_transform_edt(crop, sampling=spacing)
        sdf = np.where(crop, -(dist_in - half_voxel), dist_out - half_voxel)
        sdf = np.clip(sdf, -sdf_clip_mm, sdf_clip_mm)
        q = np.round(sdf / sdf_scale)
        # keep the sign exact at the boundary (|sdf| >= half a voxel)
        q = np.where(crop, np.minimum(q, -1), np.maximum(q, 1))
        sdf_q = q.astype(np.int8)
        crop_origin = lo.astype(np.int64)

    return ScanEntry(
        time=float(time_point),
        shape=shape,
        affine=affine,
        inverse_affine=np.linalg.inv(affine),
        center_phys=center_phys,
        spacing=spacing,
        positive_voxels=positive.astype(np.int16),
        crop_origin=crop_origin,
        sdf_q=sdf_q,
        sdf_scale=sdf_scale,
        sdf_clip_mm=float(sdf_clip_mm),
    )


_PRELOAD_TRAJECTORIES = None
_PRELOAD_CLIP = None


def _preload_worker(job):
    trajectory_index, date = job
    trj = _PRELOAD_TRAJECTORIES[trajectory_index]
    return (trj.patient_id, trj.label_id, date), build_scan_entry(trj, date, _PRELOAD_CLIP)


class ScanCache:
    """(patient, label, date) -> ScanEntry, optionally LRU-bounded."""

    def __init__(self, sdf_clip_mm=20.0, max_entries=None):
        self.sdf_clip_mm = float(sdf_clip_mm)
        self.max_entries = max_entries
        self._entries = OrderedDict()

    def __len__(self):
        return len(self._entries)

    def get(self, trj, date):
        key = (trj.patient_id, trj.label_id, str(date))
        entry = self._entries.get(key)
        if entry is None:
            entry = build_scan_entry(trj, date, self.sdf_clip_mm)
            self._entries[key] = entry
            if self.max_entries is not None and len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        elif self.max_entries is not None:
            self._entries.move_to_end(key)
        return entry

    def preload(self, trajectories, processes=8):
        """Build entries for every date of every (unique) trajectory in parallel."""
        global _PRELOAD_TRAJECTORIES, _PRELOAD_CLIP
        unique = list({lesion_key(t): t for t in trajectories}.values())
        jobs = [
            (i, d)
            for i, t in enumerate(unique)
            for d in all_dates(t)
            if (t.patient_id, t.label_id, d) not in self._entries
        ]
        if not jobs:
            return
        _PRELOAD_TRAJECTORIES, _PRELOAD_CLIP = unique, self.sdf_clip_mm
        try:
            if processes and processes > 1:
                with mp.get_context("fork").Pool(processes) as pool:
                    results = pool.imap_unordered(_preload_worker, jobs, chunksize=4)
                    for key, entry in tqdm(results, total=len(jobs), desc="Preloading scans"):
                        self._entries[key] = entry
            else:
                for job in tqdm(jobs, desc="Preloading scans"):
                    key, entry = _preload_worker(job)
                    self._entries[key] = entry
        finally:
            _PRELOAD_TRAJECTORIES, _PRELOAD_CLIP = None, None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _split_counts(total, fractions):
    counts = [int(total * f) for f in fractions]
    counts[-1] += total - sum(counts)
    return counts


def _pick(rng, pool, count):
    if count <= 0 or len(pool) == 0:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(pool, dtype=np.int64)[rng.integers(0, len(pool), size=count)]


def _box(rng, lo, hi, count):
    if count <= 0:
        return np.empty((0, 3), dtype=np.int64)
    return rng.integers(lo, hi, size=(count, 3))


def _jittered_model_coords(entry, voxels, rng):
    jitter = rng.uniform(-0.5, 0.5, size=voxels.shape)
    return entry.voxels_to_model(voxels + jitter)


def sample_history_points(entry, n_points, rng, ring_mm=3.0, fractions=(0.4, 0.3, 0.3)):
    """[n_points, 4] = (x, y, z, label): lesion, boundary ring, local box."""
    if entry.sdf_q is None:
        voxels = _box(rng, np.zeros(3, dtype=np.int64), np.asarray(entry.shape), n_points)
    else:
        n_pos, n_ring, n_box = _split_counts(n_points, fractions)
        lo, hi = entry.crop_box
        voxels = np.concatenate([
            _pick(rng, entry.positive_voxels, n_pos),
            _pick(rng, entry.ring_voxels(ring_mm), n_ring),
            _box(rng, lo, hi, n_box),
        ])
        if len(voxels) < n_points:
            voxels = np.concatenate([voxels, _box(rng, lo, hi, n_points - len(voxels))])
    labels = entry.label_at_voxels(voxels)
    xyz = _jittered_model_coords(entry, voxels, rng)
    return np.concatenate([xyz, labels[:, None]], axis=1).astype(np.float32)


def sample_query_points(target, last, n_points, rng, ring_mm=3.0,
                        fractions=(0.30, 0.30, 0.15, 0.10, 0.15)):
    """Query points at the target scan.

    Groups: target lesion, target boundary ring, ring around the last history
    lesion (growth/shrink zone), box around both lesions, whole volume.
    Returns xyz [n, 3], labels [n], sdf to the last history lesion [n] (mm).
    """
    n_pos, n_ring, n_last, n_box, n_vol = _split_counts(n_points, fractions)
    boxes = [e.crop_box for e in (target, last) if e.sdf_q is not None]
    if boxes:
        lo = np.min([b[0] for b in boxes], axis=0)
        hi = np.max([b[1] for b in boxes], axis=0)
    else:
        lo, hi = np.zeros(3, dtype=np.int64), np.asarray(target.shape)
    voxels = np.concatenate([
        _pick(rng, target.positive_voxels, n_pos),
        _pick(rng, target.ring_voxels(ring_mm), n_ring),
        _pick(rng, last.ring_voxels(ring_mm), n_last),
        _box(rng, lo, hi, n_box),
    ])
    n_vol = n_points - len(voxels)   # empty groups fall back to whole-volume samples
    voxels = np.concatenate([
        voxels, _box(rng, np.zeros(3, dtype=np.int64), np.asarray(target.shape), n_vol)
    ])
    labels = target.label_at_voxels(voxels)
    xyz = _jittered_model_coords(target, voxels, rng)
    sdf = last.sdf_at_model(xyz)
    return xyz, labels, sdf


def history_arrays(entries, n_points, rng, ring_mm=3.0):
    return {
        "points": np.stack([sample_history_points(e, n_points, rng, ring_mm) for e in entries]),
        "times": np.asarray([e.time for e in entries], dtype=np.float32),
        "log_volume": np.asarray([np.log1p(e.volume_mm3) for e in entries], dtype=np.float32),
    }


def collate_history(histories):
    """List of history_arrays dicts -> padded tensors with a validity mask."""
    batch = len(histories)
    max_len = max(len(h["times"]) for h in histories)
    n_points, n_feat = histories[0]["points"].shape[1:]
    points = torch.zeros(batch, max_len, n_points, n_feat)
    times = torch.zeros(batch, max_len)
    log_volume = torch.zeros(batch, max_len)
    mask = torch.zeros(batch, max_len, dtype=torch.bool)
    for b, h in enumerate(histories):
        length = len(h["times"])
        points[b, :length] = torch.from_numpy(h["points"])
        times[b, :length] = torch.from_numpy(h["times"])
        log_volume[b, :length] = torch.from_numpy(h["log_volume"])
        mask[b, :length] = True
    return {"points": points, "times": times, "log_volume": log_volume, "mask": mask}


def build_item(cache, trj, history_dates, target_date, history_points, query_points,
               rng, ring_mm, embedding_id):
    if str(target_date) in {str(d) for d in history_dates}:
        raise ValueError(f"Target {target_date} must not be part of the history")
    entries = [cache.get(trj, d) for d in history_dates]
    target = cache.get(trj, target_date)
    xyz, labels, sdf = sample_query_points(target, entries[-1], query_points, rng, ring_mm)
    coords = np.concatenate([xyz, np.full((len(xyz), 1), target.time, np.float32)], axis=1)
    return {
        "history": history_arrays(entries, history_points, rng, ring_mm),
        "coords": coords.astype(np.float32),
        "labels": labels.astype(np.float32),
        "sdf": sdf.astype(np.float32),
        "embedding_id": np.int64(embedding_id),
    }


def point_lstm_collate(batch):
    out = {
        "history": collate_history([item["history"] for item in batch]),
        "coords": torch.from_numpy(np.stack([item["coords"] for item in batch])),
        "labels": torch.from_numpy(np.stack([item["labels"] for item in batch])).unsqueeze(-1),
        "sdf": torch.from_numpy(np.stack([item["sdf"] for item in batch])).unsqueeze(-1),
        "patient_idx": torch.as_tensor([int(item["embedding_id"]) for item in batch]),
    }
    if "days_from_endpoint" in batch[0]:
        out["days_from_endpoint"] = torch.as_tensor(
            [float(item["days_from_endpoint"]) for item in batch]
        )
    return out


def batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class _PointLSTMBase(Dataset):
    def __init__(self, trajectories, history_points=1024, query_points=3000,
                 max_history_len=5, sdf_clip_mm=20.0, ring_mm=3.0,
                 cache=None, **_unused):
        self.history_points = history_points
        self.query_points = query_points
        self.max_history_len = max_history_len
        self.ring_mm = ring_mm
        self.cache = cache if cache is not None else ScanCache(sdf_clip_mm)
        self.validation_samples = load_validation_samples()
        with open(DatasetPaths().patient_to_idx(), encoding="utf-8") as file:
            self.patient_to_idx = json.load(file)
        self.trajectories = list(trajectories)
        self._rng = None
        self._rng_pid = None

    def get_embedding_id(self, trj):
        return np.int64(int(self.patient_to_idx[trj.patient_id]) * 100 + int(trj.label_id))

    def rng(self):
        # One generator per process, seeded from torch so runs are reproducible
        # (DataLoader gives every worker base_seed + worker_id).
        if self._rng is None or self._rng_pid != os.getpid():
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._rng_pid = os.getpid()
        return self._rng


class PointLSTMDataset(_PointLSTMBase):
    """Training: random target among the training dates, random earlier history."""

    def __init__(self, trajectories, consecutive_history_probability=0.5, **kwargs):
        super().__init__(trajectories, **kwargs)
        self.consecutive_history_probability = consecutive_history_probability
        self._dates = {}
        kept = []
        for trj in self.trajectories:
            key = lesion_key(trj)
            if key not in self._dates:
                self._dates[key] = training_dates(trj, self.validation_samples)
            if len(self._dates[key]) >= 2:
                kept.append(trj)
        dropped = len(self.trajectories) - len(kept)
        if dropped:
            print(f"PointLSTMDataset: dropped {dropped} items with < 2 training scans")
        self.trajectories = kept

    def __len__(self):
        return len(self.trajectories)

    def sample_dates(self, dates, rng):
        k = int(rng.integers(1, len(dates)))                  # target index
        if rng.random() < self.consecutive_history_probability:
            j = k - 1                                         # last history scan
        else:
            j = int(rng.integers(0, k))                       # skip scans -> longer gap
        n_history = int(rng.integers(1, self.max_history_len + 1))
        earlier = dates[:j]
        n_extra = min(n_history - 1, len(earlier))
        chosen = sorted(rng.choice(len(earlier), size=n_extra, replace=False)) if n_extra else []
        return [earlier[i] for i in chosen] + [dates[j]], dates[k]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        rng = self.rng()
        history_dates, target_date = self.sample_dates(self._dates[lesion_key(trj)], rng)
        return build_item(
            self.cache, trj, history_dates, target_date, self.history_points,
            self.query_points, rng, self.ring_mm, self.get_embedding_id(trj),
        )


class PointLSTMValidationDataset(_PointLSTMBase):
    """Held-out interpolation or extrapolation targets with deterministic sampling."""

    def __init__(self, trajectories, split, seed=0, **kwargs):
        super().__init__(trajectories, **kwargs)
        if split not in ("interpolation", "extrapolation"):
            raise ValueError(split)
        self.split = split
        self.seed = seed
        self.cases = []
        for trj in {lesion_key(t): t for t in self.trajectories}.values():
            sample = self.validation_samples.get(lesion_key(trj))
            if sample is None:
                continue
            if split == "interpolation":
                index = interpolation_index(sample)
                if index is not None:
                    self.cases.append((trj, all_dates(trj)[index], -1.0))
            else:
                for days, date in zip(sample["extrapolation_T_minus"], sample["extrapolation_dates"]):
                    self.cases.append((trj, str(date), float(days)))

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        trj, target_date, days = self.cases[idx]
        rng = np.random.default_rng(
            self.seed + zlib.crc32(f"{lesion_key(trj)}|{target_date}".encode())
        )
        history = evaluation_history(trj, target_date, self.validation_samples, self.max_history_len)
        item = build_item(
            self.cache, trj, history, target_date, self.history_points,
            self.query_points, rng, self.ring_mm, self.get_embedding_id(trj),
        )
        item["days_from_endpoint"] = days
        return item
