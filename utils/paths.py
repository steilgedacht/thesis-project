"""
Centralized path construction for the Yale Brain Mets longitudinal dataset.

Why this exists
----------------
In the original monolithic file, several classes had to instantiate *other*
classes purely to read a derived path off of them, e.g.:

    self.path = os.path.join(MRI_Dataloader(fast_load=True).data_prediction_path, ...)

That's not a real functional dependency -- it's a dependency on a string.
`DatasetPaths` knows the on-disk layout (PRE_POST_YBML / YBML / predictions
folders and their naming conventions) and nothing else. Every other class
takes a `DatasetPaths` instance (or builds one from a root) instead of
reaching into a sibling class for a path.

This class has NO imports from data_sample / mri_dataloader / patient /
lesion_trajectory, so it can never be part of an import cycle.
"""

import os

DEFAULT_DATA_PATH = "data/entire_yale_dataset/PRE_POST_YBML"


class DatasetPaths:
    def __init__(self, data_path: str = DEFAULT_DATA_PATH):
        self.data_path = data_path
        self.prediction_root = data_path.replace("PRE_POST_YBML", "predictions")
        self.ybml_root = data_path.replace("PRE_POST_YBML", "YBML")

    # ---- roots ----------------------------------------------------------
    def sample_dir(self, patient_id: str) -> str:
        return os.path.join(self.data_path, patient_id)

    def prediction_dir(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_root, patient_id, date)

    # ---- per-sample files -------------------------------------------------
    def original_sample_path(self, patient_id: str, date: str, filename: str) -> str:
        return os.path.join(self.ybml_root, patient_id, date, filename)

    def registered_transform_path(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_dir(patient_id, date), "registered_transform.npy")

    def nnunet_prediction_path(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_dir(patient_id, date), "seg_nnUnet.nii.gz")

    def lesion_segmentation_path(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_dir(patient_id, date), "label2.nii.gz")

    def lesion_trajectory_seg_path(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_dir(patient_id, date), "trajectory.nii.gz")

    def zoomed_segmentation_path(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_dir(patient_id, date), "zoomed_label2.nii.gz")

    def zoomed_pre_post_path(self, patient_id: str, date: str) -> str:
        return os.path.join(self.prediction_dir(patient_id, date), "zoomed_mri2.nii.gz")

    # ---- trajectory files -------------------------------------------------
    def lesion_trajectory_npz_path(self, patient_id: str, label_id) -> str:
        return os.path.join(self.prediction_root, patient_id, f"lesion_trajectories_{label_id}.npz")

    def lesion_trajectory_glob(self, patient_id: str) -> str:
        return os.path.join(self.prediction_root, patient_id, "lesion_trajectories_*.npz")

    def all_post_scans_glob(self) -> str:
        return self.data_path + "/**/**/*POST.nii.gz"

    def other_timepoint_glob(self, patient_id: str, date: str) -> str:
        return os.path.join(self.data_path, patient_id, date, "**_POST.nii.gz")

    # ---- dataloader samples ------------------------------------------------
    def patient_to_idx(self):
        return "experiments/05_data_visualizer/patient_to_idx.json"

    def validation_samples(self):
        return "experiments/05_data_visualizer/validation_lesions.json"

    def plotting_samples(self):
        return "experiments/05_data_visualizer/plotting_lesions.json"