import glob
import os
import pandas as pd
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from IPython.display import HTML
import plotly.express as px
from sklearn.cluster import KMeans
from skimage.measure import find_contours
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R
from plotly.colors import qualitative
import plotly.graph_objects as go
import itertools
from sklearn.cluster import AffinityPropagation
from skimage.measure import marching_cubes
import torch

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
        self.lesion_segmentation_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["label.npz"])
        self.lesion_trajectory_path = os.path.join(*self.pre_post_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["trajectory.npz"])

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

    def load_mri(self):
        """Loads the pre post Niabel file """
        return nib.load(self.pre_post_path).get_fdata()

    def load_nnUNet_prediction(self):
        """Loads the output file from the nnUNet prediction"""
        if os.path.exists(self.lesion_prediction_nnUnet_path):
            return nib.load(self.lesion_prediction_nnUnet_path).get_fdata()
        else:
            raise FileNotFoundError(f"nnUNet prediction file not found at {self.lesion_prediction_nnUnet_path}")

    def load_mri_segmentation(self):
        """Loads the lesion segmentation file"""
        if os.path.exists(self.lesion_segmentation_path):
            segmentation = np.load(self.lesion_segmentation_path)
            self.num_lesions = segmentation["num_features"].item()
            self.lesion_sizes = segmentation["lesion_sizes"]
            self.lesion_coords = segmentation["lesion_positions"]
            self.relative_lesion_sizes = segmentation["relative_lesion_sizes"]
            return segmentation["labeled_array"]
        else:
            print(f"Lesion segmentation file not found at {self.lesion_segmentation_path}, processing sample to create it.")
            return self.process_sample()
        
    def load_lesion_trajectory_segmentation(self):
        data = np.load(self.lesion_trajectory_path)
        return data["labeled_array"], data["num_features"]

    def get_other_timepoint(self, date):
        glob_path = os.path.join(self.base_path, self.patient_id, date, f"**_POST.nii.gz")
        return DataSample(glob.glob(glob_path, recursive=True)[0])

    def process_sample(self):
        mri_image = self.load_mri()
        nnUNet_prediction = self.load_nnUNet_prediction()

        # we introduce bleeding, to connect nearby lesions
        nnUNet_prediction_dilated = ndimage.binary_dilation(nnUNet_prediction, iterations=5)
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
        sizes = ndimage.sum(nnUNet_prediction, labeled_array, range(1, num_features + 1))

        # we save the position of the lesions in euclidean roation coordinates
        lesion_positions = ndimage.center_of_mass(nnUNet_prediction, labeled_array, range(1, num_features + 1))

        # because different mri scans which might not be registered have different scalings, we need to account for the size
        total_area = np.sum((mri_image > 0).astype(np.float32))

        self.num_lesions = num_features
        self.lesion_sizes = sizes
        self.relative_lesion_sizes = sizes / total_area
        self.lesion_coords = lesion_positions

        np.savez(self.lesion_segmentation_path, num_features=num_features, lesion_sizes=sizes, relative_lesion_sizes=sizes / total_area, lesion_positions=lesion_positions, labeled_array=labeled_array)

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


class Lesion_Trajectory():
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

    def iterate_patients(self):
        for patient_id in self.patient_ids:
            yield Patient(patient_id, dataloader=self)
    
    def iterate_trajectories(self):
        for trajectory_path in self.lesion_trajectory_paths:
            yield Lesion_Trajectory(load_from_trajectory_path=trajectory_path)
            

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

    def merge_lesion_to_trajectory(self):
        vol_shape = self.samples[0].load_mri().shape
        volume = self.samples[0].load_mri()
        volume = np.sum(volume[volume > 0])

        labeled_mask_cache = []
        sum_mask = np.zeros(vol_shape, dtype=np.uint8)
        for i, sample in enumerate(self.samples):
            labeled_mask = self.registrator.rigid_transform(
                sample.load_mri_segmentation(), 
                self.registered_transforms[i], 
                vol_shape
            )   

            labeled_mask_cache.append(labeled_mask)
            sum_mask = sum_mask + labeled_mask

        sum_mask = (sum_mask > 0).astype(np.uint8)            
        # we introduce a little bit of bleeding, to connect nearby lesions
        sum_mask_dialated = ndimage.binary_dilation(sum_mask, iterations=1).astype(np.uint8)
        sum_mask_dialated = ndimage.binary_dilation(sum_mask_dialated, iterations=3, axes=(0,1)).astype(np.uint8)
        labeled_lesion_mask, num_features = ndimage.label(sum_mask_dialated)
        labeled_lesion_mask = sum_mask * labeled_lesion_mask

        sizes = np.zeros((len(self.samples), num_features), dtype=np.uint32)
        # for every lesion trajectory
        for i in range(1, num_features + 1):
            lesion_mask = (labeled_lesion_mask == i).astype(np.uint8)

            sample_ids = []
            # we check if the lesion is present in each sample
            for j in range(len(self.samples)):
                lesion_mask_sample = lesion_mask * labeled_mask_cache[j]
                if np.sum(lesion_mask_sample) > 0:
                    sample_ids.append(j)
                    sizes[j][i-1] = np.sum(lesion_mask_sample)
                    labeled_mask_cache[j][lesion_mask_sample > 0] = i
            

            trajectory = Lesion_Trajectory(patient_id=self.patient_id, label_id=i, sample_ids=sample_ids, sizes=np.log(sizes[:,i-1][sizes[:,i-1] != 0] / volume))
            trajectory.save_lesion_trajectory()

        for i, sample in enumerate(self.samples):
            np.savez(
                sample.lesion_trajectory_path, 
                num_features=num_features, 
                labeled_array=labeled_mask_cache[i],
                sizes=np.log(sizes[i]/volume),
            )
    
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



# if __name__ == "__main__":
#     dataloader = MRI_Dataloader()
#     for i in dataloader:
#         print(i)
#         break

