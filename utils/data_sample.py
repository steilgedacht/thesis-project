"""
DataSample: a single pre/post scan on disk, plus everything derived from it
(registration transform, nnUNet prediction, lesion segmentation, ...).

Depends only on `DatasetPaths` -- never on MRI_Dataloader, Patient, or
Lesion_Trajectory, so importing this module can never trigger a cycle.
"""

import glob
import os
import json

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import zoom, label, binary_fill_holes
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from IPython.display import HTML

from .paths import DatasetPaths


class DataSample:
    def __init__(self, pre_post_path, load_meta_data=False, data_path: str = None):
        # build the paths
        self.base_path = os.path.join(*(pre_post_path.split(os.path.sep)[:-3]))
        self.pre_post_path = pre_post_path
        self.original_sample_path = pre_post_path.replace("PRE_POST_YBML", "YBML")

        # `self.base_path` IS the dataset root (e.g. ".../PRE_POST_YBML"), so we
        # can build a DatasetPaths from it directly instead of asking some other
        # class (e.g. MRI_Dataloader) for its path attributes.
        self.paths = DatasetPaths(data_path or self.base_path)

        # get the meta information
        self.patient_id = pre_post_path.split(os.path.sep)[-3]
        self.date = pre_post_path.split(os.path.sep)[-2]
        self.date_time = "_".join(self.pre_post_path.split("_")[-3:-1])

        self.lesion_prediction_nnUnet_path = self.paths.nnunet_prediction_path(self.patient_id, self.date)
        self.lesion_prediction_nnUnet_path_2 = self.paths.nnunet_prediction_path(self.patient_id, self.date, model=2)
        self.lesion_segmentation_path = self.paths.lesion_segmentation_path(self.patient_id, self.date)
        self.lesion_trajectory_path = self.paths.lesion_trajectory_seg_path(self.patient_id, self.date)
        self.zoomed_segmentation_path = self.paths.zoomed_segmentation_path(self.patient_id, self.date)
        self.zoomed_pre_post_path = self.paths.zoomed_pre_post_path(self.patient_id, self.date)

        self.all_patient_dates = self._get_all_dates()
        self.n_scans = len(self.all_patient_dates)

        # load everything that is in the excel sheet
        self._excel_cache = {}
        if load_meta_data:
            self._load_meta_data()

    def _extract_sequence_data(self, meta_data_parameters_filtered, sequence_class):
        columns = ["sequence_tags", "slice_thickness (mm)", "spacing_between_slices (mm)",
                   "repetition_time (ms)", "echo_time (ms)", "inversion_time (ms)"]
        row = meta_data_parameters_filtered[meta_data_parameters_filtered["sequence_class"] == sequence_class]
        if row.empty:
            return {}
        return dict(zip(columns, row[columns].values[0]))

    def _load_meta_data(self):
        excel_file = "data/Yale-Brain-Mets-Longitudinal_ClinicalData_20250605.xlsx"
        if excel_file not in self._excel_cache:
            self._excel_cache[excel_file] = {
                "Clinical_data": pd.read_excel(excel_file, sheet_name="Clinical_data"),
                "Acquisition_data": pd.read_excel(excel_file, sheet_name="Acquisition_data"),
                "image_acquisition_parameters": pd.read_excel(excel_file, sheet_name="image_acquisition_parameters"),
            }

        cache = self._excel_cache[excel_file]
        meta_data_clinical = cache["Clinical_data"]
        meta_data_acquistion = cache["Acquisition_data"]
        meta_data_parameters = cache["image_acquisition_parameters"]

        meta_data_clinical_filtered = meta_data_clinical[
            (meta_data_clinical["patient_id"] == self.patient_id) & (meta_data_clinical["study_datetime"] == self.date_time)]
        meta_data_acquistion_filtered = meta_data_acquistion[
            (meta_data_acquistion["patient_id"] == self.patient_id) & (meta_data_acquistion["study_datetime"] == self.date_time)]
        meta_data_parameters_filtered = meta_data_parameters[
            (meta_data_parameters["patient_id"] == self.patient_id) & (meta_data_parameters["study_datetime"] == self.date_time)]

        self.age_at_Imaging = meta_data_clinical_filtered["age_at_Imaging (years)"].values.item()
        self.sex = meta_data_clinical_filtered["sex"].values.item()

        self.vendor = meta_data_acquistion_filtered["vendor"].values.item()
        self.model = meta_data_acquistion_filtered["model"].values.item()
        self.field_strength = meta_data_acquistion_filtered["field_strength (T)"].values.item()
        self._2D_3D_aquisition = meta_data_acquistion_filtered["2D_3D_acquisition"].values.item()
        self.scanner_site = meta_data_acquistion_filtered["scanner_site"].values.item()

        self.pre_included = bool(meta_data_acquistion_filtered["pre_included (1=present; 0=absent)"].values.item())
        self.post_included = bool(meta_data_acquistion_filtered["post_included (1=present; 0=absent)"].values.item())
        self.t2_included = bool(meta_data_acquistion_filtered["t2_included (1=present; 0=absent)"].values.item())
        self.flair_included = bool(meta_data_acquistion_filtered["flair_included (1=present; 0=absent)"].values.item())

        self.pre_meta_data = self._extract_sequence_data(meta_data_parameters_filtered, "PRE")
        self.post_meta_data = self._extract_sequence_data(meta_data_parameters_filtered, "POST")
        self.t2_meta_data = self._extract_sequence_data(meta_data_parameters_filtered, "T2")
        self.flair_meta_data = self._extract_sequence_data(meta_data_parameters_filtered, "FLAIR")

    def _get_all_dates(self):
        base_path = os.path.join(*(self.pre_post_path.split(os.path.sep)[:-2]))
        all_dates = os.listdir(base_path)
        return all_dates

    def get_path(self):
        return self.pre_post_path

    def load_mri(self, zoomed: bool = False, affine: bool = False):
        """Loads the pre post Nibabel file"""
        path = self.zoomed_pre_post_path if zoomed else self.pre_post_path

        if affine:
            img = nib.load(path)
            return img.get_fdata(), img.affine
        return nib.load(path).get_fdata()

    def load_nnUNet_prediction(self, model: int = 1):
        """Loads the output file from the nnUNet prediction"""
        path = self.lesion_prediction_nnUnet_path if model == 1 else self.lesion_prediction_nnUnet_path_2

        if os.path.exists(path):
            return nib.load(path).get_fdata()
        else:
            raise FileNotFoundError(f"nnUNet prediction file not found at {path}")

    def load_mri_segmentation(self, affine=False, zoomed=False):
        path = self.zoomed_segmentation_path if zoomed else self.lesion_segmentation_path
        segmentation = nib.load(path)
        metadata = {}
        for ext in segmentation.header.extensions:
            if ext.get_code() == 44:
                content = ext.get_content().decode('utf-8')
                metadata = json.loads(content)
                break

        self.num_lesions = metadata.get("num_features", None)
        self.lesion_sizes = metadata.get("lesion_sizes", None)
        self.lesion_coords = metadata.get("lesion_positions", None)
        self.relative_lesion_sizes = metadata.get("relative_lesion_sizes", None)

        if affine:
            return segmentation.get_fdata(), segmentation.affine
        return segmentation.get_fdata()

    def load_lesion_trajectory_segmentation(self):
        data = np.load(self.lesion_trajectory_path)
        return data["labeled_array"], data["num_features"]

    def get_other_timepoint(self, date):
        glob_path = self.paths.other_timepoint_glob(self.patient_id, date)
        return DataSample(glob.glob(glob_path, recursive=True)[0])

    def process_sample(self):
        mri_image, affine = self.load_mri(affine=True)
        nnUNet_prediction = self.load_nnUNet_prediction()
        nnUNet_prediction2 = self.load_nnUNet_prediction(model=2)

        for z in range(nnUNet_prediction.shape[2]):
            nnUNet_prediction[:, :, z] = binary_fill_holes(nnUNet_prediction[:, :, z] > 0) + binary_fill_holes(nnUNet_prediction2[:, :, z] > 0)

        # now we label connected lesions
        labeled_array, num_features = ndimage.label(nnUNet_prediction)

        # we calculate the sizes of each lesion and the real volumen in mm³
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        lesion_sizes = ndimage.sum_labels(nnUNet_prediction, labeled_array, range(1, num_features + 1))
        volume_mm3 = lesion_sizes * np.prod(spacing)

        # we save the position of the lesions in euclidean rotation coordinates
        lesion_positions = ndimage.center_of_mass(nnUNet_prediction, labeled_array, range(1, num_features + 1))

        # because different mri scans which might not be registered have different scalings, we need to account for the size
        total_area = np.sum((mri_image > 0).astype(np.float32))

        self.num_lesions = num_features
        self.lesion_sizes = lesion_sizes
        self.relative_lesion_sizes = lesion_sizes / total_area
        self.lesion_coords = lesion_positions

        img = nib.Nifti1Image(labeled_array, affine)

        metadata = {
            "num_features": num_features,
            "lesion_sizes": lesion_sizes.tolist(),
            "volume_mm3": volume_mm3.tolist(),
            "relative_lesion_sizes": (lesion_sizes / total_area).tolist(),
            "lesion_positions": lesion_positions
        }
        json_str = json.dumps(metadata)
        extension = nib.nifti1.Nifti1Extension(44, json_str.encode('utf-8'))
        img.header.extensions.append(extension)
        nib.save(img, self.lesion_segmentation_path)

        return labeled_array

    def plot_seg_comparison(self,
                            output_path='animation.gif',
                            save_animation=False,
                            plot_mri=True,
                            plot_nnUNet_prediction=True,
                            plot_lesion_segmentation=True,
                            jupyter_mode=True
                            ):

        mri_image = self.load_mri()
        nnUNet_prediction = self.load_nnUNet_prediction() > 0

        tmp_path = self.lesion_prediction_nnUnet_path
        self.lesion_prediction_nnUnet_path = self.lesion_prediction_nnUnet_path.replace("seg_nnUnet.nii.gz", "seg_nnUnet_2.nii.gz")
        nnUNet_prediction2 = self.load_nnUNet_prediction() > 0
        self.lesion_prediction_nnUnet_path = tmp_path

        number_of_plots = sum([plot_mri, plot_nnUNet_prediction, plot_lesion_segmentation])
        fig, axes = plt.subplots(1, number_of_plots, figsize=(6 * number_of_plots, 6))
        fig.tight_layout()

        plot_counter = 0

        cmap = cm.get_cmap('Dark2', self.num_lesions + 1)
        colors = cmap(np.linspace(0, 1, self.num_lesions + 1))
        colors[0, -1] = 0
        seg_cmap = ListedColormap(colors)

        if plot_mri:
            im_mri_1 = axes[plot_counter].imshow(mri_image[:, :, 0], cmap='gray', vmin=mri_image.min(), vmax=mri_image.max(), alpha=0.5)
            im_seg_1 = axes[plot_counter].imshow(nnUNet_prediction[:, :, 0], cmap='Reds', alpha=0.3, vmin=0, vmax=1)

            axes[plot_counter].set_title('MRI')
            plot_counter += 1

        if plot_nnUNet_prediction:
            im_mri_2 = axes[plot_counter].imshow(mri_image[:, :, 0], cmap='gray', vmin=mri_image.min(), vmax=mri_image.max(), alpha=0.5)
            im_seg_2 = axes[plot_counter].imshow(nnUNet_prediction2[:, :, 0], cmap='Reds', alpha=0.3, vmin=0, vmax=1)
            axes[plot_counter].set_title('nnUNet Prediction')
            plot_counter += 1

        if plot_lesion_segmentation:
            im_mri_3 = axes[plot_counter].imshow(mri_image[:, :, 0], cmap='gray', vmin=mri_image.min(), vmax=mri_image.max(), alpha=0.5)
            im_seg_3 = axes[plot_counter].imshow(nnUNet_prediction[:, :, 0] ^ nnUNet_prediction2[:, :, 0], cmap="Reds", alpha=0.3, vmin=0, vmax=1)
            axes[plot_counter].set_title('Lesion Segmentation')

        def update(frame):
            if plot_mri:
                im_mri_1.set_data(mri_image[:, :, frame])
                im_seg_1.set_data(nnUNet_prediction[:, :, frame])
            if plot_nnUNet_prediction:
                im_mri_2.set_data(mri_image[:, :, frame])
                im_seg_2.set_data(nnUNet_prediction2[:, :, frame])
            if plot_lesion_segmentation:
                im_mri_3.set_data(mri_image[:, :, frame])
                im_seg_3.set_data(nnUNet_prediction[:, :, frame] ^ nnUNet_prediction2[:, :, frame])

        ani = animation.FuncAnimation(fig, update, frames=mri_image.shape[2], repeat=False)
        if save_animation:
            ani.save(output_path, fps=5)
        plt.close()

        if jupyter_mode:
            return HTML(ani.to_jshtml())


    def plot_mri_animation(self,
                            output_path='animation.gif',
                            save_animation=False,
                            plot_mri=True,
                            plot_nnUNet_prediction=True,
                            plot_lesion_segmentation=True,
                            jupyter_mode=True
                            ):

        mri_image = self.load_mri()
        nnUNet_prediction = self.load_nnUNet_prediction() > 0
        segmentation_labeled = self.load_mri_segmentation()

        number_of_plots = sum([plot_mri, plot_nnUNet_prediction, plot_lesion_segmentation])
        fig, axes = plt.subplots(1, number_of_plots, figsize=(6 * number_of_plots, 6))
        fig.tight_layout()

        plot_counter = 0

        if plot_mri:
            im_mri_1 = axes[plot_counter].imshow(mri_image[:, :, 0], cmap='gray', vmin=mri_image.min(), vmax=mri_image.max())
            axes[plot_counter].set_title('MRI')
            plot_counter += 1

        if plot_nnUNet_prediction:
            im_mri_2 = axes[plot_counter].imshow(mri_image[:, :, 0], cmap='gray', vmin=mri_image.min(), vmax=mri_image.max())
            im_seg_2 = axes[plot_counter].imshow(nnUNet_prediction[:, :, 0], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
            axes[plot_counter].set_title('nnUNet Prediction')
            plot_counter += 1

        if plot_lesion_segmentation:
            cmap = cm.get_cmap('Dark2', self.num_lesions + 1)
            colors = cmap(np.linspace(0, 1, self.num_lesions + 1))
            colors[0, -1] = 0
            seg_cmap = ListedColormap(colors)

            im_mri_3 = axes[plot_counter].imshow(mri_image[:, :, 0], cmap='gray', vmin=mri_image.min(), vmax=mri_image.max(), alpha=0.5)
            im_seg_3 = axes[plot_counter].imshow(segmentation_labeled[:, :, 0], cmap=seg_cmap, alpha=0.7, vmin=-0.5, vmax=self.num_lesions + 0.5)
            axes[plot_counter].set_title('Lesion Segmentation')

        def update(frame):
            if plot_mri:
                im_mri_1.set_data(mri_image[:, :, frame])
            if plot_nnUNet_prediction:
                im_mri_2.set_data(mri_image[:, :, frame])
                im_seg_2.set_data(nnUNet_prediction[:, :, frame])
            if plot_lesion_segmentation:
                im_mri_3.set_data(mri_image[:, :, frame])
                im_seg_3.set_data(segmentation_labeled[:, :, frame])

        ani = animation.FuncAnimation(fig, update, frames=mri_image.shape[2], repeat=False)
        if save_animation:
            ani.save(output_path, fps=5)
        plt.close()

        if jupyter_mode:
            return HTML(ani.to_jshtml())


    def __repr__(self):
        return f"MRI({self.patient_id}, {self.date}, {self.n_scans} scans)"

    def calculate_contours(self):
        contours_3d = []

        for z in range(self.load_mri().shape[-1]):
            slice_2d = self.load_mri()[:, :, z]

            contours = find_contours(slice_2d, level=0.5)

            for contour in contours:
                y = contour[:, 0]
                x = contour[:, 1]
                z_coords = np.full_like(x, z)

                contours_3d.append((x, y, z_coords))

        return contours_3d

    def save_registered_images(self):
        if self.registered_transform is not None:
            np.save(self.registered_transform_path, self.registered_transform)

    def zoom(self, target_shape=(500, 500, 50)):
        image, affine = self.load_mri(affine=True)
        labels = self.load_mri_segmentation()
        mask = image > 0
        coords = np.argwhere(mask)
        y0, x0, z0 = coords.min(axis=0)
        y1, x1, z1 = coords.max(axis=0) + 1

        cropped_img = image[y0:y1, x0:x1, z0:z1]
        cropped_lab = labels[y0:y1, x0:x1, z0:z1]

        factors = [t / o for t, o in zip(target_shape, cropped_img.shape)]

        scaled_image = zoom(cropped_img, factors[:-1] + [1], order=1)
        scaled_image = zoom(scaled_image, [1, 1] + [factors[-1]], order=0)

        scaled_labels = zoom(cropped_lab, factors, order=0)

        nib.save(nib.Nifti1Image(scaled_labels, affine), self.zoomed_segmentation_path)
        nib.save(nib.Nifti1Image(scaled_image, affine), self.zoomed_pre_post_path)