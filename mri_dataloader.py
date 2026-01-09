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
from scipy.optimize import minimize
from scipy.ndimage import shift
from skimage.measure import find_contours
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as R
from plotly.colors import qualitative
import plotly.graph_objects as go
import itertools
from sklearn.cluster import AffinityPropagation
from scipy.ndimage import center_of_mass

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
            self.sizes = segmentation["sizes"]
            self.lesion_rel_coords = segmentation["lesion_rel_coords"]
            return segmentation["labeled_array"]
        else:
            raise FileNotFoundError(f"Lesion segmentation file not found at {self.lesion_segmentation_path}")

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

        # calculate global center of mass of the MRI to have a reference point that is similar also to the next scan of the same patient
        mri_mask = (mri_image > 0).astype(np.float32)
        global_center_of_mass = ndimage.center_of_mass(mri_mask)

        # calculate the rotation coordinates of each lesion with respect to the global center of mass
        lesion_rel_coords = np.array([
            [pos[0] - global_center_of_mass[0],
            pos[1] - global_center_of_mass[1],
            pos[2] - global_center_of_mass[2]]
            for pos in lesion_positions
        ])


        self.num_lesions = num_features
        self.lesion_sizes = sizes
        self.lesion_rel_coords = lesion_rel_coords
        self.lesion_abs_coords = lesion_positions

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

class MRI_Dataloader:
    def __init__(self, data_path='data/entire_yale_dataset/PRE_POST_YBML'):

        globs = glob.glob(data_path + "/**/**/*POST.nii.gz", recursive=True)
        self.pre_post_samples = list(set(sorted(globs))) 
        self.data_path = data_path

        self.patient_ids = sorted(list(set([path.split(os.path.sep)[-3] for path in self.pre_post_samples])))

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

class Patient:
    def __init__(self, patient_id, dataloader=MRI_Dataloader()):
        self.patient_id = patient_id
        self.dataloader = dataloader
        self.samples = self.dataloader.find_by_patient_id(patient_id)

    def _iou_score(self, fixed, moving):
        intersection = np.logical_and(fixed, moving).sum()
        union = np.logical_or(fixed, moving).sum()
        return intersection / union if union > 0 else 0.0
    
    def make_mask(self, vol):
        return (vol > 0).astype(np.uint8)

    def rigid_transform(self, mask, params, output_shape):
        tx, ty, tz, rx, ry, rz, sx, sy, sz = params

        rot = R.from_euler("xyz", [rx, ry, rz]).as_matrix()

        scale_matrix = np.diag([sx, sy, sz])
        affine = (rot @ scale_matrix).T

        center = np.array(mask.shape) / 2

        offset = center - affine @ center - np.array([tx, ty, tz])

        transformed = affine_transform(
            mask,
            affine,
            offset=offset,
            output_shape=output_shape,
            order=1,
            mode="constant",
            cval=0
        )

        return transformed

    # def _register_image(self, reference, moving_mask):
    #     initital_params = np.array([0,0,0,0,0,0,1.0,1.0,1.0])
        
    #     def objective(params):
    #         transformed = self.rigid_transform(
    #             moving_mask,
    #             params,
    #             reference.shape
    #         )

    #         return - self._iou_score(reference, transformed)
        
    #     # Optimize translation
    #     result = minimize(
    #         objective,
    #         initital_params,
    #         method="Powell",
    #         options=dict(maxiter=15, disp=True)
    #     )
        
    #     return result.x

    def _register_image(self, reference, moving_mask):
        ref_com = center_of_mass(reference)
        mov_com = center_of_mass(moving_mask)

        tz_init = ref_com[0] - mov_com[0]
        ty_init = ref_com[1] - mov_com[1]
        tx_init = ref_com[2] - mov_com[2]
        
        initial_params = np.array([tx_init, ty_init, tz_init, 0, 0, 0, 1.0, 1.0, 1.0])
        
        def objective(params):
            transformed = self.rigid_transform(
                moving_mask,
                params,
                reference.shape
            )
            reg = 0.01 * np.sum((params[6:9] - 1.0)**2)
            return -self._iou_score(reference, transformed) + reg

        
        print(f"Starting Powell with CoM Offset: {initial_params[:3].round(2)}")
        
        result = minimize(
            objective,
            initial_params,
            method="Powell",
            options=dict(
                maxiter=30,  
                xtol=1e-3, 
                ftol=1e-3,
                disp=True
            )
        )
        
        return result.x


    def register_all_to_first(self):
        """Register all images to the first image using linear translation only."""
        reference_image = self.samples[0].load_mri()
        
        self.samples[0].registered_transform = np.array([0,0,0,0,0,0,1.0,1.0,1.0])
        self.samples[0].save_registered_images()

        registered_transforms = []
        
        for i, sample in enumerate(self.samples[1:], start=1):
            moving_image = sample.load_mri()
            
            transformation = self._register_image(self.make_mask(reference_image), self.make_mask(moving_image))
            sample.registered_transform = transformation
            sample.save_registered_images()

            print(f"Sample {i}: registered_transform = {transformation}")

            registered_transforms.append(transformation)
        
        return registered_transforms

    def plot_3d_lesion_position(self, log_size=True):
        D_point, D_sizes, D_time, D_time_absolute = [], [], [], []
        data = {}

        for i, mri in enumerate(self.samples):
            mri.load_mri_segmentation()
            
            data[mri.date] = {
                "num_features": mri.num_lesions,
                "sizes": mri.sizes,
                "lesion_rel_coords": mri.lesion_rel_coords,
            }

            for n in range(mri.lesion_rel_coords.shape[0]):
                D_point.append(mri.lesion_rel_coords[n])
                D_sizes.append(np.log(mri.sizes[n]))
                D_time.append(i)
                D_time_absolute.append(mri.date)

        maximum_n = max([len(data[d]["sizes"]) for d in data.keys()])

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


    def plot_image_registration(self, registereed_parameters):
        colors = (qualitative.Dark24)
        color_cycle = itertools.cycle(colors)

        fig = go.Figure()

        for i, sample in enumerate(self.samples):   
            col = next(color_cycle)

            points_transformed = self.rigid_transform(
                sample.load_mri(),
                registereed_parameters[i],
                self.samples[0].load_mri().shape
            )

            points_transformed = points_transformed > 0.0


            contours_3d = []

            for z in range(points_transformed.shape[-1]):
                slice_2d = points_transformed[:,:,z]

                contours = find_contours(slice_2d, level=0.5)

                for contour in contours:
                    y = contour[:, 0]
                    x = contour[:, 1]
                    z_coords = np.full_like(x, z)

                    contours_3d.append((x,y,z_coords))

            for x, y, z in contours_3d:
                fig.add_trace(go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(width=2, color=col),
                    opacity=0.6
                ))
            
        fig.update_layout(
            height=800, 
            width=900
        )

        fig.show()

    def transform_point_to_fixed(self, params, moving_volume_shape, point):
        tx, ty, tz, rx, ry, rz, sx, sy, sz = params
        
        # 1. Build the matrix in the SAME way the optimizer uses it
        # Use 'xyz' only if your array is stored as (X, Y, Z)
        # If your brain is 'sideways', you might need to try 'zyx' or swap tx/ty
        rot_mat = R.from_euler("xyz", [rx, ry, rz]).as_matrix()
        scale_mat = np.diag([sx, sy, sz])
        
        # This is the forward matrix A
        A = (rot_mat @ scale_mat).T 
        
        # Center of rotation (must be identical to the registration function)
        center = np.array(moving_volume_shape) / 2.0
        translation = np.array([tx, ty, tz])
        
        # 2. Apply the INVERSE logic
        # In Scipy: p_moving = A @ (p_fixed - center) + center - translation
        # To get p_fixed:
        A_inv = np.linalg.inv(A)
        p_moving = np.asarray(point)
        
        fixed_point = A_inv @ (p_moving - center + translation) + center
        return fixed_point



    def plot_registered_3d_lesion_position(self, registered_parameters, log_size=True):
        """
        registered_parameters: List of parameter vectors (length 9) for each sample.
                registered_parameters[0] should be the identity: [0,0,0,0,0,0,1,1,1]
        """
        D_point, D_sizes, D_time, D_time_absolute = [], [], [], []
        
        # We use Sample 0 as the reference shape
        ref_shape = self.samples[0].load_mri().shape

        for i, mri in enumerate(self.samples):
            # Ensure lesions are processed and loaded
            mri.load_mri_segmentation()
            if not hasattr(mri, 'lesion_abs_coords'):
                mri.process_sample()
                
            params = registered_parameters[i]
            vol_shape = mri.load_mri().shape

            for n in range(len(mri.lesion_abs_coords)):
                moving_point = np.array(mri.lesion_abs_coords[n])
                
                # Transform moving point to the fixed (Sample 0) coordinate system
                if i == 0:
                    fixed_point = moving_point
                else:
                    fixed_point = self.transform_point_to_fixed(params, vol_shape, moving_point)
                
                D_point.append(fixed_point)
                D_sizes.append(mri.sizes[n])
                D_time.append(i)
                D_time_absolute.append(mri.date)

        if not D_point:
            print("No lesions found.")
            return

        D_point = np.stack(D_point)
        
        # Clustering to identify the same lesion across different timepoints
        # We use the maximum number of lesions found in any single scan as n_clusters
        affprop = AffinityPropagation(random_state=0).fit(D_point)

        df = pd.DataFrame({
            "x": D_point[:, 1], 
            "y": ref_shape[0] -D_point[:, 0], 
            "z": D_point[:, 2], 
            "size": D_sizes,
            "log_size": np.log(D_sizes),
            "time": D_time,
            "date": D_time_absolute, 
            "lesion_id": affprop.labels_.astype(str) # String for discrete color map
        })

        scale_col = 'log_size' if log_size else 'size'
        
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

        # Optional: Add brain contours from the reference image (Sample 0)
        for x, y, z in self.samples[0].calculate_contours():
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=1, color="rgba(50,50,50,1)"),
                showlegend=False
            ))

        fig.update_layout(
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=300/ref_shape[0], y=300/ref_shape[1], z=1) # Adjust z based on your slice thickness
            )
        )

        fig.show()

    def __repr__(self):
        return f"Patient {self.patient_id}, {len(self.samples)} scans"

if __name__ == "__main__":
    dataloader = MRI_Dataloader()
    for i in dataloader:
        print(i)
        break

