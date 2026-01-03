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


class DataSample:
    def __init__(self, pre_post_path, load_meta_data=False):
        # build the paths
        self.base_path = os.path.join(*(pre_post_path.split(os.path.sep)[:-3]))
        self.pre_post_path = pre_post_path
        self.original_sample_path = pre_post_path.replace("PRE_POST_YBML", "YBML")

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
    

class MRI_Dataloader:
    def __init__(self, data_path='data/entire_yale_dataset/PRE_POST_YBML'):

        globs = glob.glob(data_path + "/**/**/*POST.nii.gz", recursive=True)
        self.pre_post_samples = list(set(sorted(globs))) 
        self.data_path = data_path

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

class Patient:
    def __init__(self, patient_id, dataloader=MRI_Dataloader()):
        self.patient_id = patient_id
        self.dataloader = dataloader
        self.samples = self.dataloader.find_by_patient_id(patient_id)

    def register_all_to_first(self):
        """Register all images to the first image using linear translation only."""
        reference_image = self.samples[0].load_mri()
        
        # Set correction vector for reference image
        self.samples[0].correction_vector = np.array([0.0, 0.0, 0.0])
        
        for i, sample in enumerate(self.samples[1:], start=1):
            moving_image = sample.load_mri()
            
            # Find optimal translation
            translation = self._register_image(reference_image, moving_image)
            sample.correction_vector = translation
            
            print(f"Sample {i}: correction_vector = {translation}")

    def _register_image(self, reference, moving, initial_guess=None):
        """Register moving image to reference using translation only."""
        if initial_guess is None:
            initial_guess = np.array([0.0, 0.0, 0.0])
        
        def neg_cross_correlation(translation):
            """Negative cross-correlation (for minimization)."""
            shifted = shift(moving, translation, order=1, mode='constant', cval=0)
            
            # Ensure both images have the same shape
            ref_shape = np.array(reference.shape)
            shift_shape = np.array(shifted.shape)
            min_shape = np.minimum(ref_shape, shift_shape)
            
            # Crop to overlapping region (take from center)
            ref_slices = tuple([slice(int((ref_shape[i] - min_shape[i]) / 2), int((ref_shape[i] - min_shape[i]) / 2) + int(min_shape[i]))  for i in range(len(ref_shape))])
            shift_slices = tuple([slice(int((shift_shape[i] - min_shape[i]) / 2), int((shift_shape[i] - min_shape[i]) / 2) + int(min_shape[i])) for i in range(len(shift_shape))])
            
            ref_cropped = reference[ref_slices].flatten()
            shifted_cropped = shifted[shift_slices].flatten()
            
            # Remove zero values to avoid bias
            mask = (ref_cropped > 0) & (shifted_cropped > 0)
            if mask.sum() < 10:
                return 1e6
            
            corr = np.corrcoef(ref_cropped[mask], shifted_cropped[mask])[0, 1]
            if np.isnan(corr):
                return 1e6
            return -corr  # Negative because we minimize
        
        
        # Optimize translation
        result = minimize(
            neg_cross_correlation, 
            initial_guess,
            method='Powell',
            options={'maxiter': 500}
        )
        
        return result.x

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

        # TODO find a better clustering approach
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

    def plot_image_registration(self):
        mri_image = self.load_mri()
        target_image = DataSample(self.all_patient_dates[0]) 

        

if __name__ == "__main__":
    dataloader = MRI_Dataloader()
    for i in dataloader:
        print(i)
        break

