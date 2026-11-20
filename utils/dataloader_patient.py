from datetime import datetime
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset

from .dataloader import (
    LesionDataset,
    Validation_Extrapolation_LesionDataset,
    Validation_Interpolation_LesionDataset,
    Plotting_LesionDataset,
)
from .paths import DatasetPaths


class PatientDataset(LesionDataset):
    """Sample a patient-level union of the patient's lesion masks.

    The dataset groups the supplied trajectories by patient, selects a
    training date, and overlays every supplied lesion available on that date.
    Background samples exclude each absent lesion's union across its scans.
    Items have the same ``(coords, labels, patient_idx)`` format as
    :class:`LesionDataset`.
    """

    def __init__(
        self,
        trajectories,
        device="cuda",
        dialation_iterations=5,
        background_samples=3000,
        oversample_positive_samples=True,
    ):
        super().__init__(
            trajectories,
            device=device,
            dialation_iterations=dialation_iterations,
            background_samples=background_samples,
            oversample_positive_samples=oversample_positive_samples,
        )

        grouped = {}
        for trajectory in trajectories:
            grouped.setdefault(trajectory.patient_id, []).append(trajectory)

        self.patient_trajectories = {}
        for patient_id, patient_trajectories in grouped.items():
            unique_trajectories = {}
            for trajectory in patient_trajectories:
                unique_trajectories.setdefault(trajectory.label_id, trajectory)
            self.patient_trajectories[patient_id] = list(unique_trajectories.values())

        self.patient_ids = sorted(self.patient_trajectories.keys())
        self._lesion_union_cache: dict[
            tuple[str, int], tuple[np.ndarray, np.ndarray]
        ] = {}

    def __len__(self):
        return len(self.patient_ids)

    def get_embedding_id(self, trj):
        return np.int64(self.patient_to_idx[trj.patient_id])

    def _training_dates(self, patient_id, trajectories):
        training_dates = set()
        held_out_dates = set()

        for trj in trajectories:
            dates = list(trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates)
            lesion_id = f"{trj.patient_id}_{trj.label_id}"
            if lesion_id in self.validation_samples:
                validation = self.validation_samples[lesion_id]
                extrapolation_dates = set(validation["extrapolation_dates"])
                dates = [date for date in dates if date not in extrapolation_dates]
                held_out_dates.update(extrapolation_dates)
                interpolation_index = int(validation["interpolation"])
                held_out_dates.add(dates.pop(interpolation_index))
            training_dates.update(str(date) for date in dates)

        dates = sorted(training_dates - held_out_dates)
        if not dates:
            raise ValueError(
                f"Patient {patient_id} has no training dates after validation "
                "dates are excluded."
            )
        return dates

    def _load_lesion_union(self, trj) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (trj.patient_id, trj.label_id)
        if cache_key in self._lesion_union_cache:
            return self._lesion_union_cache[cache_key]

        lesion_union = None
        reference_affine = None
        for date in sorted({str(date) for date in trj.dates}):
            mask_path = Path(trj.path).parent / date / "trajectory.nii.gz"
            image = nib.load(str(mask_path))
            lesion_mask = np.isin(image.get_fdata(), [trj.label_id])

            if lesion_union is None:
                lesion_union = lesion_mask
                reference_affine = image.affine
                continue

            if lesion_mask.shape != lesion_union.shape:
                raise ValueError(
                    f"Label shape mismatch for lesion {cache_key} on {date}."
                )
            if reference_affine is None or not np.allclose(
                image.affine, reference_affine
            ):
                raise ValueError(
                    f"Label affine mismatch for lesion {cache_key} on {date}."
                )
            np.logical_or(lesion_union, lesion_mask, out=lesion_union)

        if lesion_union is None:
            raise ValueError(f"No lesion masks found for lesion {cache_key}.")

        if reference_affine is None:
            raise ValueError(f"No lesion affine found for lesion {cache_key}.")

        self._lesion_union_cache[cache_key] = (lesion_union, reference_affine)
        return lesion_union, reference_affine

    def _load_patient_labels(
        self, patient_id, selected_date, trajectories
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels_union = None
        unavailable_union = None
        reference_affine = None

        for trj in trajectories:
            lesion_union, lesion_affine = self._load_lesion_union(trj)
            trajectory_dates = {str(date) for date in trj.dates}

            if unavailable_union is None:
                unavailable_union = np.zeros_like(lesion_union)
                reference_affine = lesion_affine
            elif lesion_union.shape != unavailable_union.shape:
                raise ValueError(
                    f"Label shape mismatch for patient {patient_id} on "
                    f"{selected_date}."
                )
            elif reference_affine is None or not np.allclose(
                lesion_affine, reference_affine
            ):
                raise ValueError(
                    f"Label affine mismatch for patient {patient_id} on "
                    f"{selected_date}."
                )

            if selected_date not in trajectory_dates:
                np.logical_or(unavailable_union, lesion_union, out=unavailable_union)
                continue

            mask_path = Path(trj.path).parent / selected_date / "trajectory.nii.gz"
            image = nib.load(str(mask_path))
            lesion_labels = np.isin(image.get_fdata(), [trj.label_id])

            if lesion_labels.shape != lesion_union.shape:
                raise ValueError(
                    f"Label shape mismatch for lesion "
                    f"{trj.patient_id}_{trj.label_id} on {selected_date}."
                )
            if not np.allclose(image.affine, lesion_affine):
                raise ValueError(
                    f"Label affine mismatch for lesion "
                    f"{trj.patient_id}_{trj.label_id} on {selected_date}."
                )

            if labels_union is None:
                labels_union = np.zeros_like(lesion_labels)
            elif lesion_labels.shape != labels_union.shape:
                raise ValueError(
                    f"Label shape mismatch for patient {patient_id} on "
                    f"{selected_date}."
                )
            if reference_affine is None or not np.allclose(
                image.affine, reference_affine
            ):
                raise ValueError(
                    f"Label affine mismatch for patient {patient_id} on "
                    f"{selected_date}."
                )
            np.logical_or(labels_union, lesion_labels, out=labels_union)

            if not np.any(lesion_labels):
                np.logical_or(unavailable_union, lesion_union, out=unavailable_union)

        if (
            labels_union is None
            or unavailable_union is None
            or reference_affine is None
        ):
            raise ValueError(
                f"No lesion masks found for patient {patient_id} on {selected_date}."
            )
        return labels_union, unavailable_union, reference_affine

    def _prepare_patient_data(self, patient_id, selected_date, trajectories):
        labels, unavailable_union, affine = self._load_patient_labels(
            patient_id, selected_date, trajectories
        )
        sampleable_positive = labels & ~unavailable_union
        positive_voxels = np.argwhere(sampleable_positive)

        target_positive_count = self.max_samples // 2
        target_sample_count = 2 * target_positive_count
        if len(positive_voxels) == 0:
            positive_count = 0
        elif self.oversample_positive_samples:
            positive_count = target_positive_count
        else:
            positive_count = min(len(positive_voxels), target_positive_count)

        negative_count = target_sample_count - positive_count
        sampleable_background = ~labels & ~unavailable_union
        negative_voxels = np.argwhere(sampleable_background)
        if negative_count and len(negative_voxels) == 0:
            raise ValueError(
                f"Patient {patient_id} has no sampleable background voxels on "
                f"{selected_date}."
            )

        positive_indices = positive_voxels
        if positive_count:
            positive_indices = positive_voxels[
                np.random.choice(
                    len(positive_voxels),
                    size=positive_count,
                    replace=self.oversample_positive_samples,
                )
            ]

        negative_indices = np.empty((0, 3), dtype=np.int64)
        if negative_count:
            border = np.logical_and(
                binary_dilation(labels, iterations=self.dialation_iterations),
                sampleable_background,
            )
            border_voxels = np.argwhere(border)
            border_count = min(negative_count // 2, len(border_voxels))
            if border_count:
                border_indices = border_voxels[
                    np.random.choice(len(border_voxels), size=border_count, replace=False)
                ]
            else:
                border_indices = np.empty((0, 3), dtype=np.int64)

            remaining_count = negative_count - border_count
            background_indices = negative_voxels[
                np.random.choice(
                    len(negative_voxels), size=remaining_count, replace=True
                )
            ]
            negative_indices = np.concatenate(
                [border_indices, background_indices], axis=0
            )

        voxel_indices = np.concatenate([positive_indices, negative_indices], axis=0)
        sample_labels = np.concatenate(
            [
                np.ones(len(positive_indices), dtype=np.float32),
                np.zeros(len(negative_indices), dtype=np.float32),
            ]
        )

        i, j, k = voxel_indices.T
        homogeneous_indices = np.column_stack(
            [voxel_indices, np.ones(len(voxel_indices), dtype=voxel_indices.dtype)]
        )
        physical_coords = (affine @ homogeneous_indices.T).T[:, :3]
        volume_center_voxel = (np.asarray(labels.shape) - 1) / 2.0
        volume_center_physical = (
            affine[:3, :3] @ volume_center_voxel + affine[:3, 3]
        )
        spatial_coords = (physical_coords - volume_center_physical) / 100.0

        first_date = min(
            datetime.strptime(str(date), "%Y-%m-%d")
            for trj in trajectories
            for date in trj.dates
        )
        time_point = (datetime.strptime(selected_date, "%Y-%m-%d") - first_date).days
        coords = np.column_stack(
            [spatial_coords, np.full(len(i), time_point, dtype=np.float32)]
        ).astype(np.float32)

        permutation = np.random.permutation(len(sample_labels))
        return coords[permutation], sample_labels[permutation]

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        trajectories = self.patient_trajectories[patient_id]
        dates = self._training_dates(patient_id, trajectories) # we get the training dates that are not used for validation for that patient 
        selected_date = str(np.random.choice(dates))
        coords, labels = self._prepare_patient_data(
            patient_id, selected_date, trajectories
        )
        return coords, labels, self.get_embedding_id(trajectories[0])


class PatientValidationExtrapolationDataset(Validation_Extrapolation_LesionDataset):
    """Validation dataset using the compact patient IDs used in training."""

    def get_embedding_id(self, trj):
        return np.int64(self.patient_to_idx[trj.patient_id])


class PatientValidationInterpolationDataset(Validation_Interpolation_LesionDataset):
    """Validation dataset using the compact patient IDs used in training."""

    def get_embedding_id(self, trj):
        return np.int64(self.patient_to_idx[trj.patient_id])


class PatientPlottingDataset(Plotting_LesionDataset):
    """Plotting trajectories with the compact patient IDs used in training."""

    def get_embedding_id(self, trj):
        return np.int64(self.patient_to_idx[trj.patient_id])


class PatientLesionCropValidationDataset(Dataset):
    """Evaluate densely in a target-lesion ROI, excluding other labeled lesions."""

    def __init__(
        self,
        trajectories,
        validation_mode,
        device="cuda",
        crop_margin_mm=10.0,
    ):
        self.device = device
        with open(DatasetPaths().patient_to_idx(), "r") as file:
            self.patient_to_idx = json.load(file)
        with open(DatasetPaths().validation_samples(), "r") as file:
            validation_samples = json.load(file)
        self.validation_samples = {
            f"{sample['Patient']}_{sample['Lesion']}": sample
            for sample in validation_samples
        }

        if validation_mode not in {"interpolation", "extrapolation"}:
            raise ValueError(f"Unknown validation mode: {validation_mode}")
        if not np.isfinite(crop_margin_mm) or crop_margin_mm < 0:
            raise ValueError("crop_margin_mm must be a finite non-negative value.")

        self.validation_mode = validation_mode
        self.crop_margin_mm = float(crop_margin_mm)
        self._lesion_union_cache = {}
        self._patient_first_dates = {}
        self.trajectories = []

        for trj in trajectories:
            lesion_id = f"{trj.patient_id}_{trj.label_id}"
            if lesion_id not in self.validation_samples:
                continue
            validation = self.validation_samples[lesion_id]
            if validation_mode == "interpolation":
                dates = list(
                    trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates
                )
                interpolation_index = int(validation["interpolation"])
                if not 0 <= interpolation_index < len(dates):
                    raise ValueError(
                        f"Invalid interpolation index for lesion {lesion_id}."
                    )
                self.trajectories.append(
                    (-1, str(dates[interpolation_index]), trj)
                )
            else:
                extrapolation_days = validation["extrapolation_T_minus"]
                extrapolation_dates = validation["extrapolation_dates"]
                if len(extrapolation_days) != len(extrapolation_dates):
                    raise ValueError(
                        f"Extrapolation day/date counts differ for lesion {lesion_id}."
                    )
                self.trajectories.extend(
                    (int(days), str(date), trj)
                    for days, date in zip(
                        extrapolation_days,
                        extrapolation_dates,
                    )
                )

        for patient_id in {trj.patient_id for _, _, trj in self.trajectories}:
            patient_trajectories = [
                trj for trj in trajectories if trj.patient_id == patient_id
            ]
            self._patient_first_dates[patient_id] = min(
                datetime.strptime(str(date), "%Y-%m-%d")
                for trj in patient_trajectories
                for date in trj.dates
            )

    def __len__(self):
        return len(self.trajectories)

    def _load_lesion_union(self, trj):
        cache_key = (trj.patient_id, trj.label_id)
        if cache_key in self._lesion_union_cache:
            return self._lesion_union_cache[cache_key]

        lesion_union: np.ndarray | None = None
        reference_affine: np.ndarray | None = None
        for date in sorted({str(date) for date in trj.dates}):
            image = nib.Nifti1Image.from_filename(
                str(Path(trj.path).parent / date / "trajectory.nii.gz")
            )
            image_affine = image.affine
            if image_affine is None:
                raise ValueError(
                    f"Missing affine for lesion {cache_key} on {date}."
                )
            lesion_mask = np.isin(image.get_fdata(), [trj.label_id])

            if lesion_union is None:
                lesion_union = lesion_mask
                reference_affine = image_affine
                continue
            if lesion_mask.shape != lesion_union.shape:
                raise ValueError(
                    f"Label shape mismatch for lesion {cache_key} on {date}."
                )
            if reference_affine is None:
                raise ValueError(f"Missing reference affine for lesion {cache_key}.")
            if not np.allclose(image_affine, reference_affine):
                raise ValueError(
                    f"Label affine mismatch for lesion {cache_key} on {date}."
                )
            np.logical_or(lesion_union, lesion_mask, out=lesion_union)

        if lesion_union is None or reference_affine is None:
            raise ValueError(f"No lesion masks found for lesion {cache_key}.")

        self._lesion_union_cache[cache_key] = (lesion_union, reference_affine)
        return lesion_union, reference_affine

    def __getitem__(self, idx):
        days_from_extrapolation, selected_date, trj = self.trajectories[idx]
        image = nib.Nifti1Image.from_filename(
            str(Path(trj.path).parent / selected_date / "trajectory.nii.gz")
        )
        image_affine = image.affine
        if image_affine is None:
            raise ValueError(
                f"Missing affine for lesion "
                f"{trj.patient_id}_{trj.label_id} on {selected_date}."
            )
        volume = image.get_fdata()
        target_mask = np.isin(volume, [trj.label_id])
        lesion_union, union_affine = self._load_lesion_union(trj)

        if volume.shape != lesion_union.shape:
            raise ValueError(
                f"Label shape mismatch for lesion "
                f"{trj.patient_id}_{trj.label_id} on {selected_date}."
            )
        if not np.allclose(image_affine, union_affine):
            raise ValueError(
                f"Label affine mismatch for lesion "
                f"{trj.patient_id}_{trj.label_id} on {selected_date}."
            )

        roi_mask = target_mask if np.any(target_mask) else lesion_union
        roi_voxels = np.argwhere(roi_mask)
        if len(roi_voxels) == 0:
            raise ValueError(
                f"Cannot define an evaluation crop for lesion "
                f"{trj.patient_id}_{trj.label_id}: it is absent from all scans."
            )

        spacing = np.linalg.norm(image_affine[:3, :3], axis=0)
        if np.any(spacing <= 0):
            raise ValueError(
                f"Invalid voxel spacing for lesion "
                f"{trj.patient_id}_{trj.label_id}."
            )
        padding = np.ceil(self.crop_margin_mm / spacing).astype(np.int64)
        crop_start = np.maximum(roi_voxels.min(axis=0) - padding, 0)
        crop_stop = np.minimum(
            roi_voxels.max(axis=0) + padding + 1, np.asarray(volume.shape)
        )
        crop_slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(crop_start, crop_stop)
        )

        crop_volume = volume[crop_slices]
        cropped_labels = target_mask[crop_slices]
        evaluation_mask = (crop_volume == 0) | cropped_labels
        if not np.any(evaluation_mask):
            raise ValueError(
                f"Evaluation crop for lesion "
                f"{trj.patient_id}_{trj.label_id} on {selected_date} "
                "contains no target or background voxels."
            )

        local_voxels = np.indices(cropped_labels.shape, dtype=np.int64)
        voxel_indices = local_voxels.reshape(3, -1).T + crop_start
        voxel_indices = voxel_indices[evaluation_mask.reshape(-1)]
        homogeneous_indices = np.column_stack(
            [voxel_indices, np.ones(len(voxel_indices), dtype=np.int64)]
        )
        physical_coords = (image_affine @ homogeneous_indices.T).T[:, :3]
        volume_center_voxel = (np.asarray(volume.shape) - 1) / 2.0
        volume_center_physical = (
            image_affine[:3, :3] @ volume_center_voxel + image_affine[:3, 3]
        )
        spatial_coords = (physical_coords - volume_center_physical) / 100.0

        first_date = self._patient_first_dates[trj.patient_id]
        time_point = (
            datetime.strptime(selected_date, "%Y-%m-%d") - first_date
        ).days
        coords = np.column_stack(
            [spatial_coords, np.full(len(voxel_indices), time_point)]
        ).astype(np.float32)
        labels = cropped_labels.reshape(-1)[evaluation_mask.reshape(-1)].astype(
            np.float32
        )
        voxel_volume_cm3 = abs(np.linalg.det(image_affine[:3, :3])) / 1000.0

        return (
            coords,
            labels,
            self.get_embedding_id(trj),
            days_from_extrapolation,
            voxel_volume_cm3,
        )

    def get_embedding_id(self, trj):
        return np.int64(self.patient_to_idx[trj.patient_id])


class PatientLesionCropInterpolationDataset(PatientLesionCropValidationDataset):
    def __init__(self, trajectories, **kwargs):
        super().__init__(trajectories, validation_mode="interpolation", **kwargs)


class PatientLesionCropExtrapolationDataset(PatientLesionCropValidationDataset):
    def __init__(self, trajectories, **kwargs):
        super().__init__(trajectories, validation_mode="extrapolation", **kwargs)
