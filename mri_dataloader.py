import glob
import os
import json
import pandas as pd
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from IPython.display import HTML
import plotly.express as px
from sklearn.cluster import KMeans, AffinityPropagation
from skimage.measure import find_contours, marching_cubes
from scipy.ndimage import affine_transform, zoom
from plotly.colors import qualitative
import plotly.graph_objects as go
import itertools
import torch
from datetime import datetime
from scipy.signal import savgol_filter


class Registrator:
    def __init__(self, device=None):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_3d_bbox(self, mask):
        z_idx = torch.nonzero(mask.sum(dim=(1,2)))
        y_idx = torch.nonzero(mask.sum(dim=(0,2)))
        x_idx = torch.nonzero(mask.sum(dim=(0,1)))

        bbox = {
            "z_min": z_idx[0].item(), "z_max": z_idx[-1].item(),
            "y_min": y_idx[0].item(), "y_max": y_idx[-1].item(),
            "x_min": x_idx[0].item(), "x_max": x_idx[-1].item(),
        }

        center = torch.tensor([
            (bbox["x_min"] + bbox["x_max"]) / 2.0,
            (bbox["y_min"] + bbox["y_max"]) / 2.0,
            (bbox["z_min"] + bbox["z_max"]) / 2.0,
        ], device=self.device)

        return bbox, center


    def rigid_transform(self, moving_image, params, output_shape):
        tx, ty, tz, rx, ry, rz, sx, sy, sz = params
                
        transformed = affine_transform(
            moving_image,
            [sx, sy, sz],
            offset=[tx, ty, tz],
            output_shape=output_shape,
            order=1,
            mode="constant",
            cval=0
        )
        return transformed


    def register(self, reference_mask, moving_mask):
        ref_t = torch.from_numpy(reference_mask).float().permute(2, 1, 0).to(self.device)
        mov_t = torch.from_numpy(moving_mask).float().permute(2, 1, 0).to(self.device)

        bbox_ref, center_ref = self.get_3d_bbox(ref_t)
        bbox_mov, center_mov = self.get_3d_bbox(mov_t)


        sx = (bbox_mov["x_max"] - bbox_mov["x_min"]) / (bbox_ref["x_max"] - bbox_ref["x_min"]) 
        sy = (bbox_mov["y_max"] - bbox_mov["y_min"]) / (bbox_ref["y_max"] - bbox_ref["y_min"]) 
        sz = (bbox_mov["z_max"] - bbox_mov["z_min"]) / (bbox_ref["z_max"] - bbox_ref["z_min"]) 

        tx = center_mov[0] -  center_ref[0] * sx
        ty = center_mov[1] -  center_ref[1] * sy
        tz = center_mov[2] -  center_ref[2] * sz

        init_params = torch.tensor([
            tx, ty, tz,
            0.0, 0.0, 0.0,
            sx, sy, sz
        ])

        return init_params.cpu().detach().numpy()

class DataSample:
    def __init__(self, pre_post_path, load_meta_data=False):
        # build the paths
        self.base_path = os.path.join(*(pre_post_path.split(os.path.sep)[:-3]))
        self.pre_post_path = pre_post_path
        self.original_sample_path = pre_post_path.replace("PRE_POST_YBML", "YBML")

        self.registered_transform_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["registered_transform.npy"])
        if os.path.exists(self.registered_transform_path):
            self.registered_transform = np.load(self.registered_transform_path)
        else:
            self.registered_transform = None

        self.lesion_prediction_nnUnet_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["0.nii.gz"])
        self.lesion_segmentation_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["label.nii.gz"])
        self.lesion_trajectory_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["trajectory.nii.gz"])
        self.zoomed_segmentation_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["zoomed_label.nii.gz"])
        self.zoomed_pre_post_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["zoomed_mri.nii.gz"])

        # get the meta information
        self.patient_id = pre_post_path.split(os.path.sep)[-3]
        self.date = pre_post_path.split(os.path.sep)[-2]
        self.date_time = "_".join(self.pre_post_path.split("_")[-3:-1])
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

        meta_data_clinical_filtered = meta_data_clinical[(meta_data_clinical["patient_id"] == self.patient_id) & (meta_data_clinical["study_datetime"] == self.date_time)]
        meta_data_acquistion_filtered = meta_data_acquistion[(meta_data_acquistion["patient_id"] == self.patient_id) & (meta_data_acquistion["study_datetime"] == self.date_time)]
        meta_data_parameters_filtered = meta_data_parameters[(meta_data_parameters["patient_id"] == self.patient_id) & (meta_data_parameters["study_datetime"] == self.date_time)]

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

    def load_mri(self, zoomed:bool=False, affine:bool=False):
        """Loads the pre post Niabel file """
        path = self.zoomed_pre_post_path if zoomed else self.pre_post_path 
        
        if affine:
            img = nib.load(path)
            return img.get_fdata(), img.affine 
        return nib.load(path).get_fdata()
    
    def load_nnUNet_prediction(self):
        """Loads the output file from the nnUNet prediction"""
        if os.path.exists(self.lesion_prediction_nnUnet_path):
            return nib.load(self.lesion_prediction_nnUnet_path).get_fdata()
        else:
            raise FileNotFoundError(f"nnUNet prediction file not found at {self.lesion_prediction_nnUnet_path}")

    def load_mri_segmentation(self, affine=False, zoomed=False):
        path = self.zoomed_segmentation_path if zoomed else self.lesion_segmentation_path
        segmentation = nib.load(path)
        metadata = {}
        for ext in segmentation.header.extensions:
            if ext.get_code() == 44:
                # Den Byte-String dekodieren und zurück in ein Dictionary umwandeln
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

        # """Loads the lesion segmentation file"""
        # if os.path.exists(self.lesion_segmentation_path):
        #     segmentation = np.load(self.lesion_segmentation_path)
        #     self.num_lesions = segmentation["num_features"].item()
        #     self.lesion_sizes = segmentation["lesion_sizes"]
        #     self.lesion_coords = segmentation["lesion_positions"]
        #     self.relative_lesion_sizes = segmentation["relative_lesion_sizes"]
        #     return segmentation["labeled_array"]
        # else:
        #     print(f"Lesion segmentation file not found at {self.lesion_segmentation_path}, processing sample to create it.")
        #     return self.process_sample()
        
    def load_lesion_trajectory_segmentation(self):
        data = np.load(self.lesion_trajectory_path)
        return data["labeled_array"], data["num_features"]

    def get_other_timepoint(self, date):
        glob_path = os.path.join(self.base_path, self.patient_id, date, f"**_POST.nii.gz")
        return DataSample(glob.glob(glob_path, recursive=True)[0])

    def process_sample(self):
        mri_image, affine = self.load_mri(affine=True)
        nnUNet_prediction = self.load_nnUNet_prediction()

        # we introduce bleeding, to connect nearby lesions, that might be fragmented
        nnUNet_prediction_dilated = ndimage.binary_dilation(nnUNet_prediction, iterations=4)
        # now we label connected lesions
        labeled_array, num_features = ndimage.label(nnUNet_prediction_dilated)
        # we want to have the original lesion size again so we multiply with the original seg
        labeled_array = nnUNet_prediction * labeled_array

        # now we want to fill out holes in each lesion, we do that with a convex hull
        for i in range(num_features):
            lesion = labeled_array == (i + 1)
            convex_hull = convex_hull_image(lesion)
            labeled_array[convex_hull] = i + 1

        # we calculate the sizes of each lesion
        lesion_sizes = ndimage.sum(nnUNet_prediction, labeled_array, range(1, num_features + 1))

        # we save the position of the lesions in euclidean roation coordinates
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
            "relative_lesion_sizes": (lesion_sizes / total_area).tolist(),
            "lesion_positions": lesion_positions
        }
        json_str = json.dumps(metadata)
        extension = nib.nifti1.Nifti1Extension(44, json_str.encode('utf-8'))
        img.header.extensions.append(extension)
        nib.save(img, self.lesion_segmentation_path)

        return labeled_array

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
        fig, axes = plt.subplots(1, number_of_plots, figsize=(6*number_of_plots, 6))
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
            # create a custom colormap so that each lesion has its own color
            cmap = cm.get_cmap('Dark2', self.num_lesions + 1)
            colors = cmap(np.linspace(0, 1, self.num_lesions + 1))
            colors[0, -1] = 0  # Set alpha of color 0 to 0
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
            slice_2d = self.load_mri()[:,:,z]

            contours = find_contours(slice_2d, level=0.5)

            for contour in contours:
                y = contour[:, 0]
                x = contour[:, 1]
                z_coords = np.full_like(x, z)

                contours_3d.append((x,y,z_coords))
        
        return contours_3d

    def save_registered_images(self):
        if self.registered_transform is not None:
            np.save(self.registered_transform_path, self.registered_transform)
    
    def load_registered_transform(self):
        if os.path.exists(self.registered_transform_path):
            self.registered_transform = np.load(self.registered_transform_path)
        else:
            print(f"Registered transform file not found at {self.registered_transform_path}. Using identity transform.")
            self.registered_transform = np.array([0,0,0,0,0,0,1.0,1.0,1.0])
        return self.registered_transform

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
        scaled_image = zoom(scaled_image, [1,1] + [factors[-1]], order=0)

        scaled_labels = zoom(cropped_lab, factors, order=0)

        nib.save(nib.Nifti1Image(scaled_labels, affine), self.zoomed_segmentation_path)
        nib.save(nib.Nifti1Image(scaled_image, affine), self.zoomed_pre_post_path)
        

class Lesion_Trajectory:
    def __init__(self, patient_id=None, label_id=None, sample_ids:list=None, sizes=None, load_from_trajectory_path=None):
        if load_from_trajectory_path is not None:
            self.path = load_from_trajectory_path
            self.load_lesion_trajectory()
        else:
            self.path = os.path.join(MRI_Dataloader(fast_load=True).data_prediction_path, patient_id, f"lesion_trajectories_{label_id}.npz")
            self.patient_id = patient_id
            self.label_id = label_id
            self.dates = []
            patient = Patient(patient_id)
            for i, sample in enumerate(patient.samples):
                if i not in sample_ids: continue
                sample.load_mri_segmentation()
                self.dates.append(sample.date)
            self.n_scans = len(sample_ids)
            self.sizes = sizes

    def save_lesion_trajectory(self):
        export = {
            "patient_id": self.patient_id,
            "label_id": self.label_id,
            "dates": self.dates,
            "n_scans": self.n_scans,
            "sizes": self.sizes,
        }
        
        np.savez(self.path, **export)

    def load_lesion_trajectory(self):
        if os.path.exists(self.path):
            data = np.load(self.path)
            self.patient_id = data["patient_id"].item()
            self.label_id = data["label_id"].item()
            self.dates = data["dates"].tolist()
            self.n_scans = data["n_scans"].item()
            self.sizes = data["sizes"]
        else:
            print(f"Lesion trajectory file not found at {self.path}. Cannot load trajectory.")

    def load_sizes(self):
        sizes = []
        patient = Patient(self.patient_id)
        for i, date in enumerate(patient.dates):
            if date not in self.dates:
                continue
            
            sample = patient.samples[i]
            data, num_features = sample.load_lesion_trajectory_segmentation()
            size = np.sum(data == self.label_id)
            sizes.append(size)
        self.sizes = sizes
        return sizes
    
    def load_labels_for_inr(self, selected_date=None, absolute_day_number=False, skip_empty=True):
        """
        Load trajectory labels for each date.
        
        Args:
            selected_date: If provided, only load this specific date
            absolute_day_number: Use absolute days or normalized time
            skip_empty: If True, skip frames with no voxels (helpful for trajectories with gaps)
        """
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.dates]
        first_date = dates[0]
        total_days = (dates[-1] - first_date).days if not absolute_day_number else 1
        days_since_first = [((d - first_date).days / total_days) for d in dates]
        if not absolute_day_number:
            days_since_first = [d * 2 - 1 for d in days_since_first]

        data = []
        skipped_count = 0
        
        for i, date in enumerate(self.dates):
            if selected_date is not None:
                date = selected_date
                i = self.dates.index(selected_date)

            path = os.path.join(*self.path.split(os.path.sep)[:-1] + [date] + ["trajectory.nii.gz"])
            if os.path.exists(path):
                data_segmentation = nib.load(path).get_fdata()
                data_segmentation = np.where(data_segmentation == self.label_id, 1, 0).astype(np.uint8)
                
                voxel_count = np.count_nonzero(data_segmentation)
                
                # Skip empty frames if requested (helps with gap-filled trajectories)
                if skip_empty and voxel_count == 0:
                    skipped_count += 1
                    if selected_date is None:  # Only skip if loading full trajectory
                        continue
                    # If specific date requested but empty, still return it
                    data.append((data_segmentation, days_since_first[i]))
                    if selected_date is not None:
                        return data[0]
                else:
                    data.append((data_segmentation, days_since_first[i]))
                    if selected_date is not None:
                        return data[0]
            else:
                if selected_date is not None:
                    print(f"Path not found: {path}")
                continue
        
        if selected_date is None and skipped_count > 0:
            print(f"Note: Skipped {skipped_count} empty frames from trajectory")
        
        return data
    
    def plot_trajectory_sizes(self):
        fig, ax = plt.subplots()
        ax.plot(self.dates, np.exp(self.sizes))
        ax.set_xlabel("Date")
        ax.set_ylabel("Size")
        ax.set_title(f"Lesion Trajectory for Patient {self.patient_id}, Lesion {self.label_id}")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1 * np.max(np.exp(self.sizes)))
        formatter = FuncFormatter(lambda x, p: f'{x:.2e}')
        ax.yaxis.set_major_formatter(formatter)
        plt.tight_layout()
        plt.show()

    def plot_animation(self):
        fig, ax = plt.subplots()
        im = ax.imshow(np.zeros((500, 500)), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Lesion Trajectory for Patient {self.patient_id}, Lesion {self.label_id}")

        def update(frame):
            data_segmentation, _ = self.load_labels_for_inr(selected_date=self.dates[frame])
            # Take maximum projection across z-axis to convert 3D to 2D
            data_2d = np.max(data_segmentation, axis=2)
            im.set_data(data_2d)

        ani = animation.FuncAnimation(fig, update, frames=len(self.dates), repeat=False)
        plt.close()
        return HTML(ani.to_jshtml())


    def extract_growth_phase(self, sizes, smoothing_window=5, min_start_idx=1):
        """
        Extract the main growth phase of a lesion trajectory, ignoring early peaks.
        
        Args:
            sizes: array of lesion sizes (can be log-scale)
            dates: optional dates for reference
            smoothing_window: window size for Savitzky-Golay filter
            min_start_idx: minimum index to consider as peak (avoids very early spikes)
            
        Returns:
            start_idx, end_idx, plot_data
        """
        sizes_array = np.array(sizes)
        n = len(sizes_array)
        
        # Smooth the signal
        if n > smoothing_window:
            smoothed = savgol_filter(sizes_array, smoothing_window, 2)
        else:
            smoothed = sizes_array
        
        # Compute first derivative (growth rate)
        derivative = np.gradient(smoothed)
        
        # Find peak, but ignore very early peaks (skip first few points)
        valid_range_start = min(min_start_idx, n - 2)
        peak_idx = valid_range_start + np.argmax(smoothed[valid_range_start:])
        
        threshold = np.std(derivative) * 0.5
        
        # Find growth start: first index where derivative becomes significantly positive
        growth_start_candidates = np.where(derivative[:peak_idx] > threshold)[0]
        if len(growth_start_candidates) > 0:
            start_idx = growth_start_candidates[0]
        else:
            start_idx = max(0, peak_idx - 3)
        
        # Find growth end: first index after peak where derivative becomes negative
        growth_end_candidates = np.where(derivative[peak_idx:] < -threshold)[0]
        if len(growth_end_candidates) > 0:
            end_idx = peak_idx + growth_end_candidates[0]
        else:
            end_idx = min(n - 1, peak_idx + 1)
        
        # Optimize start_idx: search in window around initial start_idx for lowest value
        start_idx = max(0, min(start_idx, n - 1))
        search_start_left = max(0, start_idx - 5)
        search_start_right = min(n - 1, start_idx + 5)
        
        best_start = start_idx
        best_start_value = sizes_array[start_idx]
        
        for idx in range(search_start_left, search_start_right + 1):
            if sizes_array[idx] < best_start_value:
                best_start = idx
                best_start_value = sizes_array[idx]
        
        start_idx = best_start
        
        # Optimize end_idx: search in window around initial end_idx for highest value
        end_idx = max(0, min(end_idx, n - 1))
        search_end_left = max(0, end_idx - 5)
        search_end_right = min(n - 1, end_idx + 5)
        
        best_end = end_idx
        best_end_value = sizes_array[end_idx]
        
        for idx in range(search_end_left, search_end_right + 1):
            if sizes_array[idx] > best_end_value:
                best_end = idx
                best_end_value = sizes_array[idx]
        
        end_idx = best_end
        
        # Ensure start_idx is always before end_idx
        if start_idx >= end_idx:
            start_idx = max(0, end_idx - 1)
        
        # After enforcing start < end, further optimize:
        # Descend the start: search backwards from start_idx for lower values
        # Allow up to 5% increase before stopping
        for idx in range(start_idx - 1, max(-1, start_idx - 10), -1):
            if idx >= 0 and idx < end_idx:
                # Allow the value to be up to 5% higher than start_idx
                if sizes_array[idx] <= sizes_array[start_idx] + np.log(1.05):
                    start_idx = idx
                else:
                    break
        
        # Ascend the end: search forwards from end_idx for higher values
        # Allow up to 5% decrease before stopping
        for idx in range(end_idx + 1, min(n, end_idx + 10)):
            if idx > start_idx:
                # Allow the value to be up to 5% lower than end_idx
                if sizes_array[idx] >= sizes_array[end_idx] + np.log(0.95):
                    end_idx = idx
                else:
                    break


        return start_idx, end_idx, {'smoothed': smoothed, 'derivative': derivative, 'peak': peak_idx}

class MRI_Dataloader:
    def __init__(self, data_path='data/entire_yale_dataset/PRE_POST_YBML', fast_load=False):

        self.data_path = data_path
        self.data_prediction_path = data_path.replace("PRE_POST_YBML", "predictions")

        if not fast_load:
            globs = glob.glob(data_path + "/**/**/*POST.nii.gz", recursive=True)
            self.pre_post_samples = list(set(sorted(globs))) 
            self.patient_ids = sorted(list(set([path.split(os.path.sep)[-3] for path in self.pre_post_samples])))

            globs = glob.glob(self.data_prediction_path + "/**/lesion_trajectories_*")
            self.lesion_trajectory_paths = list(set(sorted(globs))) 
        
        self.cache_lesion_trajectories = None


    def __iter__(self):
        for i in range(self.__len__()):
            yield DataSample(self.pre_post_samples[i])

    def __len__(self):
        return len(self.pre_post_samples)

    def find_by_patient_id(self, patient_id):
        filtered_samples = [sample for sample in self.pre_post_samples if f"{os.path.sep}{patient_id}{os.path.sep}" in sample]
        return [DataSample(sample) for sample in sorted(filtered_samples)]
    
    def find_by_date(self, date):
        filtered_samples = [sample for sample in self.pre_post_samples if f"{os.path.sep}{date}{os.path.sep}" in sample]
        return [DataSample(sample) for sample in sorted(filtered_samples)]
    
    def find_by_patient_id_and_date(self, patient_id, date):
        filtered_samples = [sample for sample in self.pre_post_samples if f"{os.path.sep}{patient_id}{os.path.sep}" in sample and f"{os.path.sep}{date}{os.path.sep}" in sample]
        return DataSample(filtered_samples[0])

    def find_lesion_trajectory(self, patient_id, label_id):
        trajectory_path = os.path.join(self.data_prediction_path, patient_id, f"lesion_trajectories_{label_id}.npz")
        if os.path.exists(trajectory_path):
            return Lesion_Trajectory(load_from_trajectory_path=trajectory_path)
        else:
            print(f"Lesion trajectory file not found at {trajectory_path}. Cannot load trajectory.")
            return None

    def iterate_patients(self):
        for patient_id in self.patient_ids:
            yield Patient(patient_id, dataloader=self)
    
    def iterate_trajectories(self):
        for trajectory_path in self.lesion_trajectory_paths:
            yield Lesion_Trajectory(load_from_trajectory_path=trajectory_path)
    
    def cache_lesion_trajectories_from_n_scans(self, n_scans:int = 8, only_growing:bool = False):
        trajectories = []
        for trj in self.iterate_trajectories():
            sizes = len(trj.dates)
            if only_growing:
                start_idx, end_idx, _ = trj.extract_growth_phase(trj.sizes)
                sizes = end_idx - start_idx
                trj.allowed_dates = trj.dates[start_idx:end_idx+1]

            if sizes < n_scans: continue
            
            trajectories.append(trj)

        print(f"Found {len(trajectories)} trajectories with at least {n_scans} scans.")
        self.cache_lesion_trajectories = trajectories
            
    def plot_lesion_trajectories_line_plot(self, trajectories=None, x_ticks_real_time=False):
        if trajectories is None and self.cache_lesion_trajectories is not None:
            trajectories = self.cache_lesion_trajectories
        elif trajectories is None and self.cache_lesion_trajectories is None:
            print("No trajectories provided and no cached trajectories found. Please provide trajectories or cache them first with self.cache_lesion_trajectories_from_n_scans().")
            return
        
        fig = px.line()

        for i, trj in enumerate(trajectories):
            
            if x_ticks_real_time:
                dates = [datetime.strptime(d, "%Y-%m-%d") for d in trj.dates]
                first_date = dates[0]
                days_since_first = [(d - first_date).days for d in dates]
            else:
                days_since_first = list(range(len(trj.dates)))


            fig.add_scatter(
                x=days_since_first, 
                y=np.exp(trj.sizes)**(1/3), 
                mode='lines+markers',
                name=f'Patient: {trj.patient_id}, Lesion: {trj.label_id}',
                line=dict(color=px.colors.qualitative.Dark24[i % 24], width=1)
            )

        fig.update_layout(
            title="Lesion Size Over Time",
            xaxis_title="Time",
            yaxis_title="Size",
            yaxis_type="log",
            template="plotly_white",
            height=800
        )

        fig.show()

    def plot_lesion_trajectories_heatmap(self, trajectories=None, skip_top=0, plot_change=False, normalize_rows=False):
        """
        Plot lesion trajectories as a heatmap.
        
        Args:
            trajectories: List of trajectories to plot. Uses cached trajectories if None.
            skip_top: Number of top trajectories to skip (by max size).
            plot_change: If True, plot change rates instead of absolute sizes.
            normalize_rows: If True, normalize each row to [0, 1].
        """
        if trajectories is None and self.cache_lesion_trajectories is not None:
            trajectories = self.cache_lesion_trajectories
        elif trajectories is None:
            print("No trajectories provided and no cached trajectories found. "
                "Please provide trajectories or cache them first with self.cache_lesion_trajectories_from_n_scans().")
            return
        
        # Filter top trajectories if requested
        trajectories = self._filter_top_trajectories(trajectories, skip_top)
        
        # Prepare heatmap data
        heatmap_data, trj_list = self._prepare_heatmap_data(trajectories, plot_change, normalize_rows)
        
        # Sort by trajectory length
        sorted_indices = sorted(range(len(trj_list)), key=lambda i: trj_list[i][1])
        heatmap_data = heatmap_data[sorted_indices]
        trj_list = [trj_list[i] for i in sorted_indices]
        
        # Create labels
        labels = [f'Patient: {trj.patient_id}, Lesion: {trj.label_id} (n={length})' 
                for trj, length in trj_list]
        
        # Create and display heatmap
        self._create_heatmap(heatmap_data, labels, plot_change)

    def _filter_top_trajectories(self, trajectories, skip_top):
        """Filter out the top N trajectories by maximum size."""
        trj_with_sizes = []
        for trj in trajectories:
            y = trj.sizes[trj.sizes != 0]
            sizes = np.exp(y)**(1/3)
            trj_with_sizes.append((trj, np.max(sizes)))
        
        # Sort by size and skip top ones
        trj_with_sizes.sort(key=lambda x: x[1], reverse=True)
        return [trj for trj, _ in trj_with_sizes[skip_top:]]

    def _prepare_heatmap_data(self, trajectories, plot_change, normalize_rows):
        """Prepare data for heatmap visualization."""
        max_len = max([len(trj.sizes[trj.sizes != 0]) for trj in trajectories])
        if plot_change:
            max_len -= 1  # Change rates have one fewer point
        
        heatmap_data = []
        trj_list = []
        
        for trj in trajectories:
            y = trj.sizes[trj.sizes != 0]
            sizes = np.exp(y)**(1/3)
            
            # Calculate change rates if requested
            if plot_change:
                data = np.diff(sizes)
            else:
                data = sizes
            
            # Pad with NaN
            padded = np.full(max_len, np.nan)
            padded[:len(data)] = data
            
            # Normalize rows if requested
            if normalize_rows:
                padded = self._normalize_row(padded)
            
            heatmap_data.append(padded)
            trj_list.append((trj, len(sizes)))
        
        return np.array(heatmap_data), trj_list

    def _normalize_row(self, row):
        """Normalize a row to [0, 1]."""
        valid_mask = ~np.isnan(row)
        if not np.any(valid_mask):
            return row
        
        min_val = np.nanmin(row)
        max_val = np.nanmax(row)
        
        if max_val > min_val:
            row[valid_mask] = (row[valid_mask] - min_val) / (max_val - min_val)
        else:
            row[valid_mask] = 0.5
        
        return row

    def _create_heatmap(self, heatmap_data, labels, plot_change):
        """Create and display the heatmap."""
        max_len = heatmap_data.shape[1]
        
        if plot_change:
            color_scale = "RdBu_r"
            color_label = "Change Rate"
            max_abs = np.nanmax(np.abs(heatmap_data))
            zmin, zmax = -max_abs, max_abs
        else:
            color_scale = "Viridis"
            color_label = "Size"
            zmin, zmax = None, None
        
        fig = px.imshow(
            heatmap_data,
            labels=dict(x="Time", y="Trajectory", color=color_label),
            x=list(range(max_len)),
            y=labels,
            color_continuous_scale=color_scale,
            title=f"Lesion {'Change Rates' if plot_change else 'Size'} Over Time (Heatmap - sorted by length)",
            zmin=zmin,
            zmax=zmax
        )
        
        fig.update_layout(height=max(600, len(labels) * 15), width=900)
        fig.show()

class Patient:
    def __init__(self, patient_id, dataloader=MRI_Dataloader(), registrator=Registrator()):
        self.patient_id = patient_id
        self.dataloader = dataloader
        self.samples = self.dataloader.find_by_patient_id(patient_id)
        self.dates = [sample.date for sample in self.samples]
        self.path = os.path.join(self.dataloader.data_path, patient_id)
        self.registrator = registrator
        self.load_registered_transforms()
        self.patient_trajectory_paths = glob.glob(os.path.join(self.dataloader.data_prediction_path, patient_id, "lesion_trajectories_*.npz"))

    def register_all_to_first(self, registrator: Registrator = Registrator()):
        """Register all images to the first image using linear translation only."""
        reference_image = self.samples[0].load_mri()
        
        self.samples[0].registered_transform = np.array([0,0,0,0,0,0,1.0,1.0,1.0])
        self.samples[0].save_registered_images()

        registered_transforms = []
        
        for sample in self.samples[1:]:
            moving_image = sample.load_mri()
            
            transformation = registrator.register(reference_image, moving_image)
            sample.registered_transform = transformation
            sample.save_registered_images()

            registered_transforms.append(transformation)
            print(f"Registered {sample.date} to {self.samples[0].date} with transformation:\n{transformation}")

        return registered_transforms

    def plot_3d_lesion_position(self, log_size=True):
        D_point, D_sizes, D_time, D_time_absolute = [], [], [], []
        data = {}

        for i, mri in enumerate(self.samples):
            mri.load_mri_segmentation()
            
            data[mri.date] = {
                "num_features": mri.num_lesions,
                "lesion_sizes": mri.lesion_sizes,
                "lesion_coords": mri.lesion_coords,
            }

            for n in range(mri.lesion_coords.shape[0]):
                D_point.append(mri.lesion_coords[n])
                D_sizes.append(np.log(mri.lesion_sizes[n]))
                D_time.append(i)
                D_time_absolute.append(mri.date)

        maximum_n = max([len(data[d]["lesion_sizes"]) for d in data.keys()])

        D_point = np.stack(D_point)

        kmeans = KMeans(n_clusters=maximum_n, random_state=0, n_init=200).fit(D_point)
        labels = kmeans.labels_

        df = pd.DataFrame({
            "x": D_point[:,0], 
            "y": D_point[:,1], 
            "z": D_point[:,2], 
            "log_size":D_sizes, 
            "size":np.exp(D_sizes), 
            "time":D_time,
            "time_absolute":D_time_absolute, 
            "labels": labels,
            "patient": self.patient_id
        })
        scale = 'log_size' if log_size else 'size'
        fig = px.scatter_3d(df, x='x', y='y', z='z', size=scale, color='time', height=800, width=900, symbol=labels)
        
        fig.show()

    def contur_plot(self, fig, data, col):
        data = data > 0.0

        contours_3d = []

        for z in range(data.shape[-1]):
            slice_2d = data[:,:,z]

            contours = find_contours(slice_2d, level=0.5)

            for contour in contours:
                x = contour[:, 0]
                y = contour[:, 1]
                z_coords = np.full_like(x, z)

                contours_3d.append((x,y,z_coords))

        for x, y, z in contours_3d:
            fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="lines",line=dict(width=2, color=col), opacity=0.6))


    def plot_image_registration(self, registered_parameters = [], registrator: Registrator = Registrator()):
        colors = (qualitative.Dark24)
        color_cycle = itertools.cycle(colors)

        fig = go.Figure()

        for i, sample in enumerate(self.samples):   
            col = next(color_cycle)

            if hasattr(sample, "registered_transform") and sample.registered_transform is not None:
                rp = sample.registered_transform
            elif registered_parameters:
                rp = registered_parameters[i]
            else:
                rp = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0])
                print(f"Sample {i} has no registered transform. Using no transformation.")

            points_transformed = registrator.rigid_transform(
                sample.load_mri(),
                rp,
                self.samples[0].load_mri().shape
            )
            
            self.contur_plot(fig, points_transformed, col)

            
        fig.update_layout(
            height=800, 
            width=900
        )

        fig.show()

    def plot_registered_3d_lesion_position(self, registered_parameters=[], log_size=True, relative_size=False):
        """
        registered_parameters: List of parameter vectors (length 9) for each sample.
                registered_parameters[0] should be the identity: [0,0,0,0,0,0,1,1,1]
        """
        D_point, D_sizes, D_relative_sizes, D_time, D_time_absolute = [], [], [], [], []
        
        ref_shape = self.samples[0].load_mri().shape
        for i, mri in enumerate(self.samples):
            
            # Ensure lesions are processed and loaded
            mri.load_mri_segmentation()
            if not hasattr(mri, 'lesion_coords'):
                mri.process_sample()
            
            if hasattr(mri, "registered_transform") and mri.registered_transform is not None:
                params = mri.registered_transform
            elif registered_parameters:
                params = registered_parameters[i]
            else:
                params = np.array([0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0])
                print(f"Sample {i} has no registered transform. Using no transformation.")

            for n in range(len(mri.lesion_coords)):
                moving_point = np.array(mri.lesion_coords[n])
                
                # Transform moving point to the fixed (Sample 0) coordinate system
                fixed_point = moving_point

                if i > 0:
                    tx, ty, tz, _, _, _, sx, sy, sz = params
                    fixed_point = (moving_point - np.array([tx, ty, tz])) / np.array([sx, sy, sz]) 
                
                D_point.append(fixed_point)
                D_sizes.append(mri.lesion_sizes[n])
                D_relative_sizes.append(mri.relative_lesion_sizes[n])
                D_time.append(i)
                D_time_absolute.append(mri.date)

        if not D_point:
            print("No lesions found.")
            return

        D_point = np.stack(D_point)
        D_relative_sizes = (D_relative_sizes - min(D_relative_sizes)) / (max(D_relative_sizes) - min(D_relative_sizes) + 1e-8) * 10 
        
        # Clustering to identify the same lesion across different timepoints
        # We use the maximum number of lesions found in any single scan as n_clusters
        affprop = AffinityPropagation(random_state=0).fit(D_point)

        df = pd.DataFrame({
            "x": D_point[:, 1], 
            "y": ref_shape[0] -D_point[:, 0], 
            "z": D_point[:, 2], 
            "size": D_sizes,
            "relative_size": D_relative_sizes,
            "log_size": np.log(D_sizes),
            "time": D_time,
            "date": D_time_absolute, 
            "lesion_id": affprop.labels_.astype(str) # String for discrete color map
        })

        scale_col = 'log_size' if log_size else 'size'
        if relative_size:
            scale_col = 'relative_size'
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', 
            size=scale_col,
            size_max=30, 
            color='time',
            symbol='lesion_id',
            hover_data=['date', 'size'],
            title=f"Longitudinal Lesion Tracking: Patient {self.patient_id}",
            height=800, width=1000
        )

        # brain contours
        for x, y, z in self.samples[0].calculate_contours():
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=1, color="rgba(50,50,50,1)"),
                showlegend=False
            )
        )

        fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=300/ref_shape[0], y=300/ref_shape[1], z=1) # Adjust z based on your slice thickness
            )
        )

        fig.show()

    def __repr__(self):
        return f"Patient {self.patient_id}, {len(self.samples)} scans"
    
    def plot_average_slice_trajectory(self):

        means = []
        for i, sample in enumerate(self.samples):
            means.append(sample.load_mri().mean(axis=(0, 1)))
        
        means_interp = []
        for mean in means:
            m = mean.copy()
            
            sliced_mean = m[m != 0]

            x_old = np.linspace(0, 1, len(sliced_mean))
            x_new = np.linspace(0, 1, 100)
            mean_interp = np.interp(x_new, x_old, sliced_mean)
            mean_interp = mean_interp / np.sort(mean_interp)[-5] 
            means_interp.append(mean_interp)


        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        cmap = plt.get_cmap("cool")
        colors = cmap(np.linspace(0, 1, len(means)))
        for i, mean in enumerate(means):
            axes[0].plot(mean, label=f"Sample {i}", color=colors[i])
        axes[0].set_title("Mean MRI Intensity per Slice")
        axes[0].set_xlabel("Slice Index")
        axes[0].set_ylabel("Mean Intensity")
        axes[0].legend()

        for i, mean_interp in enumerate(means_interp):
            axes[1].plot(mean_interp, label=f"Sample {i} (Interpolated)", color=colors[i])
        axes[1].set_title("Interpolated Mean Intensity (100 pts)")
        axes[1].set_xlabel("Interpolated Index")
        axes[1].set_ylabel("Normalized Intensity")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def process_samples(self):
        for sample in self.samples:
            sample.process_sample()

    def load_registered_transforms(self):
        self.registered_transforms = []
        for sample in self.samples:
            if hasattr(sample, "registered_transform") and sample.registered_transform is not None:
                self.registered_transforms.append(sample.registered_transform)
            else:
                self.registered_transforms.append(sample.load_registered_transform())
        return self.registered_transforms

    def _transform_points(self, points, transform_params):
        """
        Apply rigid transformation to 3D points.
        
        Args:
            points: Nx3 array of 3D points
            transform_params: 9-element array [tx, ty, tz, rx, ry, rz, sx, sy, sz]
        
        Returns:
            Transformed Nx3 array
        """
        tx, ty, tz, rx, ry, rz, sx, sy, sz = transform_params
        
        # Apply scaling and translation
        transformed = points.copy()
        transformed[:, 0] = (points[:, 0] - tx) / sx
        transformed[:, 1] = (points[:, 1] - ty) / sy
        transformed[:, 2] = (points[:, 2] - tz) / sz
        
        return transformed

    def _match_lesions_between_scans(self, scan_idx, prev_centroids, prev_sizes, curr_centroids, 
                                     curr_sizes, distance_threshold=None, debug=False):
        """
        Find optimal matching between lesions in CONSECUTIVE scans using Hungarian algorithm.
        
        **Simplified approach without registration transforms:**
        - Matches based on spatial proximity in original scan coordinates
        - Uses generous distance thresholds (brain doesn't move much between scans)
        - Validates with size ratios
        
        Args:
            scan_idx: Current scan index (only for debugging)
            prev_centroids: Mx3 array of centroids from previous scan
            prev_sizes: M array of lesion sizes from previous scan
            curr_centroids: Nx3 array of centroids from current scan
            curr_sizes: N array of lesion sizes from current scan
            distance_threshold: Maximum distance for matching. If None, uses 100 pixels
            debug: If True, print diagnostic information
        
        Returns:
            List of (prev_idx, curr_idx) tuples for valid matches
        """
        if len(prev_centroids) == 0 or len(curr_centroids) == 0:
            return []
        
        if distance_threshold is None:
            distance_threshold = 100  # Generous default for consecutive scans
        
        # Compute pairwise Euclidean distances (direct, no transforms)
        distances = np.linalg.norm(
            prev_centroids[:, np.newaxis, :] - curr_centroids[np.newaxis, :, :],
            axis=2
        )
        
        if debug:
            print(f"    Scan {scan_idx-1}->{scan_idx}: distance matrix (min/max): {distances.min():.2f}/{distances.max():.2f}")
        
        # Apply size penalty: penalize matches with very different sizes
        # Be more lenient here: allow 0.3-3.0x size changes between scans
        size_ratio_matrix = np.zeros_like(distances)
        for i in range(len(prev_centroids)):
            for j in range(len(curr_centroids)):
                ratio = curr_sizes[j] / (prev_sizes[i] + 1e-6)
                # Penalize extreme size changes but don't eliminate them
                if ratio < 0.3 or ratio > 3.0:
                    size_ratio_matrix[i, j] = 100  # Moderate penalty
                elif ratio < 0.5 or ratio > 2.0:
                    size_ratio_matrix[i, j] = 10   # Mild penalty
        
        # Combined cost matrix: distance + size penalty
        cost_matrix = distances + size_ratio_matrix
        
        # Hungarian algorithm for optimal assignment
        prev_indices, curr_indices = linear_sum_assignment(cost_matrix)
        
        # Only keep matches within distance threshold and reasonable size
        matches = []
        for p_idx, c_idx in zip(prev_indices, curr_indices):
            distance = distances[p_idx, c_idx]
            size_ratio = curr_sizes[c_idx] / (prev_sizes[p_idx] + 1e-6)
            
            # Accept if within distance threshold AND size is somewhat reasonable
            if distance < distance_threshold and 0.2 <= size_ratio <= 5.0:
                matches.append((p_idx, c_idx))
                if debug:
                    print(f"      MATCH: lesion {p_idx}->{c_idx}, dist={distance:.1f}px, size_ratio={size_ratio:.2f}x")
        
        if debug:
            print(f"      -> {len(matches)} matches found\n")
        
        return matches

    def _find_continuation(self, scan_idx, lesion_centroids, lesion_sizes, 
                          scan_lesions, visited, max_gap=2, spatial_threshold=50, 
                          allow_size_change=2.0):
        """
        Find the next occurrence of a lesion, allowing gaps.
        
        Args:
            scan_idx: Current scan index
            lesion_centroids: Centroid of current lesion
            lesion_sizes: Size of current lesion
            scan_lesions: List of all scan lesion data
            visited: Set of already-visited (scan, lesion_idx) tuples
            max_gap: Maximum number of scans to look ahead
            spatial_threshold: Max distance to search
            allow_size_change: Max size ratio to allow (e.g., 2.0 = up to 2x size change)
        
        Returns:
            (next_scan, next_lesion_idx) or (None, None) if not found
        """
        # Search forward in subsequent scans, allowing gaps
        for future_scan in range(scan_idx + 1, min(scan_idx + 1 + max_gap, len(scan_lesions))):
            if len(scan_lesions[future_scan]['centroids']) == 0:
                continue
            
            # Find unvisited lesion closest to current position
            best_dist = float('inf')
            best_idx = None
            
            for lesion_idx in range(len(scan_lesions[future_scan]['labels'])):
                if (future_scan, lesion_idx) in visited:
                    continue
                
                # Check spatial proximity and size consistency
                future_centroid = scan_lesions[future_scan]['centroids'][lesion_idx]
                future_size = scan_lesions[future_scan]['sizes'][lesion_idx]
                
                dist = np.linalg.norm(lesion_centroids - future_centroid)
                size_ratio = future_size / (lesion_sizes + 1e-6)
                
                # Both distance and size must be reasonable
                # Allow more size variation when bridging gaps
                inv_ratio = (lesion_sizes + 1e-6) / future_size if future_size > 0 else float('inf')
                size_valid = (1.0/allow_size_change <= size_ratio <= allow_size_change)
                
                if dist < spatial_threshold and size_valid:
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = lesion_idx
            
            if best_idx is not None:
                return future_scan, best_idx
        
        return None, None

    def _build_trajectories_from_matches(self, scan_lesions, matches, num_scans,
                                         max_gap=2, spatial_threshold=50, 
                                         size_ratio_threshold=4.0):
        """
        Build lesion trajectories by following match chains across scans, allowing gaps.
        
        Key feature: Seeds trajectories from ALL scans, not just the first one.
        This ensures that:
        - Lesions present in scan 0 start trajectories forward
        - NEW lesions appearing in scan i (unmatched) start fresh trajectories
        - Every detected lesion is part of exactly one trajectory
        
        Trajectories can skip timepoints if a lesion is absent (labeling error, registration
        issue, or real disappearance). Only scans where the lesion is actually present are
        included in the trajectory.
        
        Args:
            scan_lesions: List of dicts with 'centroids', 'sizes', 'labels' for each scan
            matches: List of (scan_idx, prev_idx, curr_idx) tuples
            num_scans: Number of scans
            max_gap: Max scans to skip (default 2)
            spatial_threshold: Max distance for gap-filling (default 50)
            size_ratio_threshold: Allow size changes up to this factor (default 4.0)
        
        Returns:
            List of trajectory dicts: {scan_indices: [...], labels: [(scan_idx, label), ...], sizes: [...]}
        """
        # Build adjacency information (direct consecutive matches)
        adj = {}  # (scan, lesion_idx) -> (next_scan, next_lesion_idx)
        for scan_idx, prev_idx, curr_idx in matches:
            key = (scan_idx, prev_idx)
            adj[key] = (scan_idx + 1, curr_idx)
        
        # Track which (scan, lesion_idx) have been assigned to trajectories
        visited = set()
        trajectories = []
        
        # Phase 1: Seed trajectories from lesions in scan 0
        for lesion_idx in range(len(scan_lesions[0]['labels'])):
            if (0, lesion_idx) in visited:
                continue
            
            trajectory = self._build_single_trajectory(
                scan_idx=0, lesion_idx=lesion_idx,
                scan_lesions=scan_lesions, adj=adj, visited=visited,
                num_scans=num_scans, max_gap=max_gap,
                spatial_threshold=spatial_threshold,
                size_ratio_threshold=size_ratio_threshold
            )
            trajectories.append(trajectory)
        
        # Phase 2: Seed NEW trajectories from unmatched lesions in subsequent scans
        # This handles lesions that appear for the first time (new lesions)
        for scan_idx in range(1, num_scans):
            for lesion_idx in range(len(scan_lesions[scan_idx]['labels'])):
                if (scan_idx, lesion_idx) in visited:
                    continue
                
                # This lesion wasn't matched from previous scan - start a NEW trajectory
                trajectory = self._build_single_trajectory(
                    scan_idx=scan_idx, lesion_idx=lesion_idx,
                    scan_lesions=scan_lesions, adj=adj, visited=visited,
                    num_scans=num_scans, max_gap=max_gap,
                    spatial_threshold=spatial_threshold,
                    size_ratio_threshold=size_ratio_threshold
                )
                trajectories.append(trajectory)
        
        return trajectories

    def _build_single_trajectory(self, scan_idx, lesion_idx, scan_lesions, adj, visited,
                                num_scans, max_gap, spatial_threshold, size_ratio_threshold):
        """
        Build a single trajectory starting from (scan_idx, lesion_idx).
        Follows matches forward and handles gaps.
        
        Args:
            scan_idx: Starting scan index
            lesion_idx: Starting lesion index
            scan_lesions: All scan lesion data
            adj: Adjacency dict from direct matches
            visited: Set of visited (scan, lesion) pairs (will be updated)
            num_scans: Total number of scans
            max_gap, spatial_threshold, size_ratio_threshold: Gap-filling parameters
        
        Returns:
            Trajectory dict
        """
        # Initialize trajectory
        trajectory = {
            'scan_indices': [scan_idx],
            'labels': [(scan_idx, scan_lesions[scan_idx]['labels'][lesion_idx])],
            'sizes': [scan_lesions[scan_idx]['sizes'][lesion_idx]],
            'centroids': [scan_lesions[scan_idx]['centroids'][lesion_idx]]
        }
        visited.add((scan_idx, lesion_idx))
        
        current_scan = scan_idx
        current_lesion_idx = lesion_idx
        current_centroid = scan_lesions[scan_idx]['centroids'][lesion_idx]
        current_size = scan_lesions[scan_idx]['sizes'][lesion_idx]
        
        # Follow the chain forward
        while current_scan < num_scans - 1:
            # Try direct consecutive match first
            if (current_scan, current_lesion_idx) in adj:
                next_scan, next_lesion_idx = adj[(current_scan, current_lesion_idx)]
                
                trajectory['scan_indices'].append(next_scan)
                trajectory['labels'].append((next_scan, scan_lesions[next_scan]['labels'][next_lesion_idx]))
                trajectory['sizes'].append(scan_lesions[next_scan]['sizes'][next_lesion_idx])
                trajectory['centroids'].append(scan_lesions[next_scan]['centroids'][next_lesion_idx])
                
                visited.add((next_scan, next_lesion_idx))
                current_scan = next_scan
                current_lesion_idx = next_lesion_idx
                current_centroid = scan_lesions[next_scan]['centroids'][next_lesion_idx]
                current_size = scan_lesions[next_scan]['sizes'][next_lesion_idx]
            else:
                # No direct match - look ahead allowing gaps
                next_scan, next_lesion_idx = self._find_continuation(
                    current_scan, current_centroid, current_size, 
                    scan_lesions, visited, max_gap=max_gap, 
                    spatial_threshold=spatial_threshold,
                    allow_size_change=size_ratio_threshold
                )
                
                if next_scan is not None:
                    trajectory['scan_indices'].append(next_scan)
                    trajectory['labels'].append((next_scan, scan_lesions[next_scan]['labels'][next_lesion_idx]))
                    trajectory['sizes'].append(scan_lesions[next_scan]['sizes'][next_lesion_idx])
                    trajectory['centroids'].append(scan_lesions[next_scan]['centroids'][next_lesion_idx])
                    
                    visited.add((next_scan, next_lesion_idx))
                    current_scan = next_scan
                    current_lesion_idx = next_lesion_idx
                    current_centroid = scan_lesions[next_scan]['centroids'][next_lesion_idx]
                    current_size = scan_lesions[next_scan]['sizes'][next_lesion_idx]
                else:
                    # No continuation found - trajectory ends
                    break
        
        return trajectory

    def merge_lesion_to_trajectory(self, max_gap=7, spatial_threshold=40, 
                                   size_ratio_threshold=12.0, distance_threshold=40):
        """
        Match lesions across scans using lenient spatial matching.
        Uses Hungarian algorithm for optimal 1-to-1 correspondence between consecutive scans.
        
        **Approach**: Direct spatial matching without registration transforms.
        - Matches based on centroid proximity (brain position relatively stable across consecutive scans)
        - Allows generous spatial thresholds
        - Implements smart gap-filling for missing lesions at timepoints
        
        Handles realistic scenario where:
        - Some lesions disappear (merge, resolve, or labeling errors)
        - New lesions appear
        - Variable number of lesions per timepoint
        
        Args:
            max_gap: Max scans to skip when looking for lesion continuation (default 2)
            spatial_threshold: Max pixel distance for gap-filling matches (default 150)
            size_ratio_threshold: Allow size changes up to this factor (default 3.0 = 3x)
            distance_threshold: Max distance for consecutive scan matches (default 100)
        """
        vol_shape = (500, 500, 50)
        volume, affine = self.samples[0].load_mri(zoomed=True, affine=True)
        volume = np.sum(volume[volume > 0])

        # Step 1: Extract lesion data for each scan
        print(f"Extracting lesion data from {len(self.samples)} scans...")
        scan_lesions = []
        lesion_counts = []
        for i, sample in enumerate(self.samples):
            labeled_mask = sample.load_mri_segmentation(zoomed=True)
            num_lesions = int(labeled_mask.max())
            lesion_counts.append(num_lesions)
            
            if num_lesions > 0:
                centroids = ndimage.center_of_mass(
                    labeled_mask,
                    labeled_mask,
                    range(1, num_lesions + 1)
                )
                centroids = np.array(centroids)
                
                # Get sizes for each lesion
                sizes = ndimage.sum(
                    np.ones_like(labeled_mask),
                    labeled_mask,
                    range(1, num_lesions + 1)
                )
            else:
                centroids = np.empty((0, 3))
                sizes = np.array([])
            
            scan_lesions.append({
                'centroids': centroids,
                'sizes': sizes,
                'labels': np.arange(1, num_lesions + 1),
                'labeled_mask': labeled_mask
            })
            print(f"  Scan {i} ({self.samples[i].date}): {num_lesions} lesions")
        
        print(f"\nLesion counts per scan: {lesion_counts}")

        # Step 2: Match lesions between consecutive scans
        print("\nMatching lesions between consecutive scans...")
        matches = []  # List of (scan_idx, prev_idx, curr_idx)
        unmatched_by_scan = []  # Track unmatched lesions
        
        for i in range(1, len(self.samples)):
            prev_centroids = scan_lesions[i-1]['centroids']
            prev_sizes = scan_lesions[i-1]['sizes']
            curr_centroids = scan_lesions[i]['centroids']
            curr_sizes = scan_lesions[i]['sizes']
            
            # Enable debug for first 3 transitions
            debug = (i <= 3)
            
            # Find optimal matches (lenient spatial matching, no transforms)
            scan_matches = self._match_lesions_between_scans(
                i, prev_centroids, prev_sizes, curr_centroids, curr_sizes,
                distance_threshold=distance_threshold, debug=debug
            )
            
            # Track which lesions got matched
            matched_prev = set(m[0] for m in scan_matches)
            unmatched_count = len(prev_centroids) - len(matched_prev)
            
            for prev_idx, curr_idx in scan_matches:
                matches.append((i-1, prev_idx, curr_idx))
            
            print(f"  Scan {i-1} -> {i}: {len(scan_matches)}/{len(prev_centroids)} matches " +
                  f"({unmatched_count} unmatched from previous)")
            
            unmatched_by_scan.append((i, unmatched_count))

        # Step 3: Build trajectories from match chains
        print("\nBuilding trajectories from match chains...")
        trajectories = self._build_trajectories_from_matches(
            scan_lesions, matches, len(self.samples), 
            max_gap=max_gap, spatial_threshold=spatial_threshold,
            size_ratio_threshold=size_ratio_threshold
        )
        
        trajectories_with_gaps = sum(1 for t in trajectories if len(t['scan_indices']) < len(self.samples))
        print(f"Found {len(trajectories)} lesion trajectories")
        print(f"  - {trajectories_with_gaps} trajectories have gaps (missing timepoints)")
        print(f"  - {len(trajectories) - trajectories_with_gaps} trajectories span all timepoints")
        
        # Filter trajectories: only keep those with more than 2 entries
        trajectories_filtered = [t for t in trajectories if len(t['scan_indices']) > 2]
        trajectories_short = len(trajectories) - len(trajectories_filtered)
        
        if trajectories_short > 0:
            print(f"\nFiltering: Removing {trajectories_short} trajectories with ≤2 entries")
            print(f"Keeping {len(trajectories_filtered)} trajectories with >2 entries")
        
        trajectories = trajectories_filtered

        # Step 4: Create relabeled masks and save trajectories
        labeled_mask_cache = [np.zeros(vol_shape, dtype=np.uint8) for _ in range(len(self.samples))]
        trajectory_sizes = np.zeros((len(self.samples), len(trajectories)), dtype=np.uint32)
        
        for traj_id, trajectory in enumerate(trajectories):
            sample_ids = trajectory['scan_indices']
            
            # CRITICAL: Filter out empty frames BEFORE saving
            # Only keep timepoints where the lesion actually exists 
            valid_frames = []
            removed_frames = []
            
            for frame_idx, scan_idx in enumerate(sample_ids):
                orig_mask = scan_lesions[scan_idx]['labeled_mask']
                orig_label = trajectory['labels'][frame_idx][1]
                lesion_mask = (orig_mask == orig_label).astype(np.uint8)
                voxel_count = np.sum(lesion_mask)
                
                if voxel_count > 0:  # Frame has actual lesion data
                    valid_frames.append(frame_idx)
                else:
                    removed_frames.append((scan_idx, self.samples[scan_idx].date, voxel_count))
            
            # Print summary of removed frames
            if removed_frames:
                print(f"  Trajectory {traj_id+1}: Removing {len(removed_frames)} empty frames")
                for scan_idx, date, voxels in removed_frames[:3]:  # Show first 3
                    print(f"    - Scan {scan_idx} ({date}): {voxels} voxels")
                if len(removed_frames) > 3:
                    print(f"    ... and {len(removed_frames)-3} more")
            
            # If all frames are empty, skip this trajectory entirely
            if len(valid_frames) == 0:
                print(f"  Trajectory {traj_id+1}: SKIPPED - all {len(sample_ids)} frames are empty!")
                continue
            
            # Create clean trajectory with only valid frames
            if len(valid_frames) < len(sample_ids):
                cleaned_trajectory = {
                    'scan_indices': [sample_ids[i] for i in valid_frames],
                    'labels': [trajectory['labels'][i] for i in valid_frames],
                    'sizes': [trajectory['sizes'][i] for i in valid_frames],
                    'centroids': [trajectory['centroids'][i] for i in valid_frames]
                }
                sample_ids_to_save = cleaned_trajectory['scan_indices']
                print(f"    Final: {len(valid_frames)}/{len(trajectory['scan_indices'])} valid frames")
            else:
                cleaned_trajectory = trajectory
                sample_ids_to_save = sample_ids
            
            # For each scan in this trajectory, mark the lesion with the trajectory ID
            for scan_idx in sample_ids_to_save:
                # Find the original label for this scan in this trajectory
                for scan_in_traj, label in cleaned_trajectory['labels']:
                    if scan_in_traj == scan_idx:
                        orig_label = label
                        break
                
                # Copy the lesion to the new mask
                orig_mask = scan_lesions[scan_idx]['labeled_mask']
                lesion_mask = (orig_mask == orig_label).astype(np.uint8)
                labeled_mask_cache[scan_idx][lesion_mask > 0] = traj_id + 1
                trajectory_sizes[scan_idx, traj_id] = np.sum(lesion_mask)
            
            # Save trajectory (with cleaned frames only!)
            log_sizes = np.log(cleaned_trajectory['sizes'] / (volume + 1e-7) + 1e-7)
            trajectory_obj = Lesion_Trajectory(
                patient_id=self.patient_id,
                label_id=traj_id + 1,
                sample_ids=sample_ids_to_save,  # Use cleaned sample_ids
                sizes=log_sizes
            )
            trajectory_obj.save_lesion_trajectory()

        # Step 5: Save relabeled masks
        print("\nSaving relabeled masks...")
        for i, sample in enumerate(self.samples):
            img = nib.Nifti1Image(labeled_mask_cache[i], affine)
            metadata = {
                "num_features": len(trajectories),
                "sizes": np.log(trajectory_sizes[i] / (volume + 1e-7) + 1e-7).tolist()
            }
            
            json_str = json.dumps(metadata)
            extension = nib.nifti1.Nifti1Extension(44, json_str.encode('utf-8'))
            img.header.extensions.append(extension)
            nib.save(img, sample.lesion_trajectory_path)
        
        print("Trajectory building complete!")

    def diagnose_trajectory_empty_frames(self, trajectory_idx=1):
        """
        Diagnose if a trajectory has empty frames (no actual lesion data at some timepoints).
        Helpful for debugging gap-filled trajectories.
        
        Args:
            trajectory_idx: Which trajectory to examine (1-indexed from saved trajectories)
        """
        trajectories = self.load_lesion_trajectories()
        if trajectory_idx < 1 or trajectory_idx > len(trajectories):
            print(f"Invalid trajectory index. Available: 1-{len(trajectories)}")
            return
        
        trj = trajectories[trajectory_idx - 1]
        print(f"\n=== Diagnosing Trajectory {trajectory_idx} (Label {trj.label_id}) ===")
        print(f"Patient: {trj.patient_id}")
        print(f"Number of scans in trajectory: {len(trj.dates)}")
        print(f"Dates: {trj.dates}")
        print(f"Sizes (log): {trj.sizes}")
        
        # Load all data for this trajectory
        print("\nLoading trajectory data for all dates:")
        data_all = trj.load_labels_for_inr()
        
        for i, (data_seg, time_point) in enumerate(data_all):
            non_zero_voxels = np.count_nonzero(data_seg)
            print(f"  {trj.dates[i]}: {non_zero_voxels} voxels (time_point={time_point:.3f})")
            
            if non_zero_voxels == 0:
                print(f"    ^ WARNING: Frame is EMPTY!")
        
        return data_all

    
    def load_lesion_trajectories(self):
        trajectories = []
        for path in self.patient_trajectory_paths:
            trajectories.append(Lesion_Trajectory(load_from_trajectory_path=path))
        return trajectories


    def plot_lesion_shape_trajectory(self):
        fig = go.Figure()
        
        colors = px.colors.sequential.Agsunset

        vol_shape = self.samples[0].load_mri().shape
        
        for i, sample in enumerate(self.samples):
            labeled_mask = sample.load_mri_segmentation()
            params = self.registered_transforms[i]
            labeled_mask = self.registrator.rigid_transform(labeled_mask, params, vol_shape)
            
            # Determine color for this timepoint
            color_idx = int((i / len(self.samples)) * (len(colors) - 1))
            time_color = colors[color_idx]

            unique_labels = np.unique(labeled_mask)
            for label in unique_labels:
                if label == 0: continue # Skip background
                
                lesion_mask = (labeled_mask == label).astype(np.uint8)
                
                try:
                    verts, faces, _, _= marching_cubes(lesion_mask, step_size=2, allow_degenerate=True, method="lewiner")        

                    # sample random points via a barycentric sampling method
                    n_samples = 10
                    sampled_points = []

                    for _ in range(n_samples):
                        face_idx = np.random.randint(0, len(faces))
                        face = faces[face_idx]
                        
                        r1, r2 = np.random.random(2)
                        if r1 + r2 > 1:
                            r1 = 1 - r1
                            r2 = 1 - r2
                        
                        point = (1 - r1 - r2) * verts[face[0]] + r1 * verts[face[1]] + r2 * verts[face[2]]
                        sampled_points.append(point)

                    sampled_points = np.array(sampled_points)

                    fig.add_trace(go.Scatter3d(
                        x=sampled_points[:, 1],
                        y=vol_shape[0] - sampled_points[:, 0],
                        z=sampled_points[:, 2],
                        mode="markers",
                        marker=dict(size=5, color=time_color, opacity=0.3),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                except RuntimeError:
                    # Marching cubes fails if the lesion is on the very edge or too thin
                    continue

        # Add reference brain outline for context
        for x, y, z in self.samples[0].calculate_contours():
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=1, color="rgba(50,50,50,1)"),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=f"3D Lesion Shape Trajectory for Patient {self.patient_id}",
            width=1000,
            height=800
        )
        
        fig.show()

