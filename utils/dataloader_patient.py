from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from .dataloader import (
    LesionDataset,
    Validation_Extrapolation_LesionDataset,
    Validation_Interpolation_LesionDataset,
    Plotting_LesionDataset,
)


class PatientDataset(LesionDataset):
    """Sample a patient-level union of the patient's lesion masks.

    The dataset groups the supplied trajectories by patient, selects a
    training date, and overlays every supplied lesion available on that date.
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

    @staticmethod
    def _load_patient_labels(patient_id, selected_date, trajectories):
        labels_union = None
        reference_affine = None

        for trj in trajectories:
            if selected_date not in {str(date) for date in trj.dates}:
                continue

            mask_path = Path(trj.path).parent / selected_date / "trajectory.nii.gz"
            image = nib.load(str(mask_path))
            lesion_labels = np.isin(image.get_fdata(), [trj.label_id])

            if labels_union is None:
                labels_union = lesion_labels
                reference_affine = image.affine
                continue

            if lesion_labels.shape != labels_union.shape:
                raise ValueError(
                    f"Label shape mismatch for patient {patient_id} on "
                    f"{selected_date}."
                )
            if not np.allclose(image.affine, reference_affine):
                raise ValueError(
                    f"Label affine mismatch for patient {patient_id} on "
                    f"{selected_date}."
                )
            np.logical_or(labels_union, lesion_labels, out=labels_union)

        if labels_union is None:
            raise ValueError(
                f"No lesion masks found for patient {patient_id} on {selected_date}."
            )
        return labels_union, reference_affine

    def _prepare_patient_data(self, patient_id, selected_date, trajectories):
        labels, affine = self._load_patient_labels(
            patient_id, selected_date, trajectories
        )
        positive_voxels = np.argwhere(labels)

        target_positive_count = self.max_samples // 2
        target_sample_count = 2 * target_positive_count
        if len(positive_voxels) == 0:
            positive_count = 0
        elif self.oversample_positive_samples:
            positive_count = target_positive_count
        else:
            positive_count = min(len(positive_voxels), target_positive_count)

        negative_count = target_sample_count - positive_count
        negative_voxels = np.argwhere(~labels)
        if negative_count and len(negative_voxels) == 0:
            raise ValueError(
                f"Patient {patient_id} has no background voxels to sample on "
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
            border = binary_dilation(
                labels, iterations=self.dialation_iterations
            ) & ~labels
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
