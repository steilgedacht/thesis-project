import numpy as np
from scipy.ndimage import binary_dilation

from .dataloader import LesionDataset


class DenseLesionDataset(LesionDataset):
    """Sample observed labels and temporal constraints from dense lesion regions.

    Each item holds out one training scan from the core/union calculation,
    supervises its real mask at its actual time, and adds intermediate-time
    samples constrained by the other scans' intersection and union.
    """

    def __init__(
        self,
        trajectories,
        device="cuda",
        dialation_iterations=5,
        background_samples=3000,
        oversample_positive_samples=True,
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
        self._positive_count_cache = {}

    def _training_dates(self, trj):
        dates = list(trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates)
        lesion_id = f"{trj.patient_id}_{trj.label_id}"
        if lesion_id in self.validation_samples:
            validation = self.validation_samples[lesion_id]
            extrapolation_dates = validation["extrapolation_dates"]
            dates = [date for date in dates if date not in extrapolation_dates]
            interpolation_index = int(validation["interpolation"])
            dates.pop(interpolation_index)
        return dates

    def _load_scan(self, trj, date):
        (labels, time_point), affine = trj.load_labels_for_inr(
            selected_date=date,
            absolute_day_number=True,
            affine=True,
        )
        return np.asarray(labels, dtype=np.uint8), float(time_point), affine

    def _get_positive_counts(self, cache_key, trj, dates, target_date):
        cached = self._positive_count_cache.get(cache_key)
        if cached is not None:
            target_labels, target_time, affine = self._load_scan(trj, target_date)
            counts, cached_affine, times = cached
            if target_labels.shape != counts.shape:
                raise ValueError(
                    f"Label shape changed for lesion {trj.patient_id}_{trj.label_id}"
                )
            return target_labels, times[target_date], cached_affine

        first_labels, first_time, affine = self._load_scan(trj, dates[0])
        count_dtype = np.uint16 if len(dates) <= np.iinfo(np.uint16).max else np.uint32
        counts = np.zeros(first_labels.shape, dtype=count_dtype)
        times = {}
        target_labels = None

        for date in dates:
            if date == dates[0]:
                labels, time_point = first_labels, first_time
            else:
                labels, time_point, scan_affine = self._load_scan(trj, date)
                if labels.shape != counts.shape:
                    raise ValueError(
                        f"Label shape changed for lesion {trj.patient_id}_{trj.label_id}"
                    )
                if not np.allclose(scan_affine, affine):
                    raise ValueError(
                        f"Affine changed for lesion {trj.patient_id}_{trj.label_id}"
                    )

            np.add(counts, labels, out=counts, casting="unsafe")
            times[date] = time_point
            if date == target_date:
                target_labels = labels

        self._positive_count_cache[cache_key] = (counts, affine, times)
        if target_labels is None:
            raise ValueError(f"Target date {target_date} is not in the training dates")
        return target_labels, times[target_date], affine

    @staticmethod
    def _sample_voxels(voxels, count, replace=True):
        if count == 0:
            return np.empty((0, 3), dtype=np.int64)
        if len(voxels) == 0:
            return np.empty((0, 3), dtype=np.int64)
        indices = np.random.choice(len(voxels), size=count, replace=replace or count > len(voxels))
        return voxels[indices]

    def _sample_observed(self, labels, sample_count):
        positive_voxels = np.argwhere(labels > 0)
        negative_voxels = np.argwhere(labels <= 0)
        target_positive_count = sample_count // 2

        if len(positive_voxels) == 0:
            positive_count = 0
        elif len(negative_voxels) == 0:
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

        border = binary_dilation(labels, iterations=self.dialation_iterations) & (labels <= 0)
        border_voxels = np.argwhere(border)
        border_count = min(negative_count // 5, len(border_voxels))
        border_samples = self._sample_voxels(
            border_voxels, border_count, replace=False
        )

        remaining_negative_count = negative_count - border_count
        negative_samples = self._sample_voxels(
            negative_voxels,
            remaining_negative_count,
            replace=True,
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

    def _sample_intermediate(self, core, outside_union, sample_count):
        core_voxels = np.argwhere(core)
        outside_voxels = np.argwhere(outside_union)
        if len(core_voxels) == 0 and len(outside_voxels) == 0:
            raise ValueError(
                "No persistent lesion or outside-union voxels available for "
                "intermediate-time supervision"
            )

        if len(core_voxels) and len(outside_voxels):
            positive_count = sample_count // 2
        elif len(core_voxels):
            positive_count = sample_count
        else:
            positive_count = 0

        negative_count = sample_count - positive_count
        positive_samples = self._sample_voxels(core_voxels, positive_count)
        negative_samples = self._sample_voxels(outside_voxels, negative_count)
        voxels = np.concatenate([positive_samples, negative_samples], axis=0)
        labels = np.concatenate(
            [
                np.ones(len(positive_samples), dtype=np.float32),
                np.zeros(len(negative_samples), dtype=np.float32),
            ]
        )
        return voxels, labels

    @staticmethod
    def _voxel_coords(voxels, affine, volume_shape, times):
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

        cache_key = (trj.patient_id, trj.label_id, tuple(dates))
        target_date = str(np.random.choice(dates))
        target_labels, target_time, affine = self._get_positive_counts(
            cache_key, trj, dates, target_date
        )
        positive_counts, _, times = self._positive_count_cache[cache_key]

        remaining_scan_count = len(dates) - 1
        remaining_positive_counts = positive_counts - target_labels
        core = remaining_positive_counts == remaining_scan_count
        union = remaining_positive_counts > 0

        intermediate_count = self.max_samples // 3
        observed_count = self.max_samples - intermediate_count

        observed_voxels, observed_labels = self._sample_observed(
            target_labels, observed_count
        )
        intermediate_voxels, intermediate_labels = self._sample_intermediate(
            core, ~union, intermediate_count
        )

        min_time = min(times[date] for date in dates)
        max_time = max(times[date] for date in dates)
        intermediate_times = np.random.uniform(
            min_time, max_time, size=intermediate_count
        ).astype(np.float32)

        coords = np.concatenate(
            [
                self._voxel_coords(
                    observed_voxels,
                    affine,
                    target_labels.shape,
                    np.full(observed_count, target_time, dtype=np.float32),
                ),
                self._voxel_coords(
                    intermediate_voxels,
                    affine,
                    target_labels.shape,
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
