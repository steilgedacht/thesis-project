import json
import numpy as np
from .paths import DatasetPaths
from torch.utils.data import Dataset
from scipy.ndimage import binary_dilation

class LesionDataset(Dataset):
    def __init__(self, 
                 trajectories, 
                 device='cuda', 
                 dialation_iterations=5, 
                 background_samples=3000):

        self.device = device
        self.trajectories = trajectories

        with open(DatasetPaths().patient_to_idx(), "r") as f:
            self.patient_to_idx = json.load(f)

        with open(DatasetPaths().validation_samples(), "r") as f:
            self.validation_samples = json.load(f)
            self.validation_patients = [sample["Patient"] + "_" + sample["Lesion"]  for sample in self.validation_samples]

        self.dialation_iterations = dialation_iterations
        self.max_samples = background_samples
        self.proportion_dialation_background = 5
        self.proportion_pos_neg_samples = 2


    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trj = self.trajectories[idx]
            
        dates_list = trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates

        if f"{trj.patient_id}_{trj.label_id}" in self.validation_patients:
            dates_list = dates_list[:-1]
            half_of_the_list = len(dates_list) // 2
            del dates_list[half_of_the_list]

        random_time_point = str(np.random.choice(dates_list))

        coords, labels_sampled = self.prepare_data(random_time_point, trj)

        return coords, labels_sampled, self.get_embedding_id(trj)

    def get_embedding_id(self, trj):
        return np.int64((self.patient_to_idx[trj.patient_id] * 100) + trj.label_id)

    def prepare_data(self, random_time_point, trj):
        (labels, time_point), affine = trj.load_labels_for_inr(selected_date=random_time_point, absolute_day_number=True, affine=True)
        
        pos_coords = np.argwhere(labels)

        border_samples = binary_dilation(labels, iterations=self.dialation_iterations) - labels
        border_samples_coords = np.argwhere(border_samples)

        num_neg_needed = self.max_samples // self.proportion_pos_neg_samples
        neg_coords = border_samples_coords.tolist()

        if len(neg_coords) > num_neg_needed // self.proportion_dialation_background:
            neg_coords = [neg_coords[i] for i in np.random.choice(len(neg_coords), size=num_neg_needed // 2, replace=True).tolist()]

        while len(neg_coords) < num_neg_needed:
            candidate_coords = np.array([
                np.random.randint(0, s, num_neg_needed) for s in labels.shape
            ]).T
            is_bg = labels[candidate_coords[:,0], candidate_coords[:,1], candidate_coords[:,2]] <= 0.5
            neg_coords.extend(candidate_coords[is_bg])
        
        neg_coords = np.array(neg_coords)[:num_neg_needed]
        
        if len(pos_coords) == 0:
            all_sampled_indices = np.concatenate([neg_coords, neg_coords], axis=0)
        else:
            pos_idx = np.random.choice(len(pos_coords), size=self.max_samples//2, replace=True)
            pos_sampled = pos_coords[pos_idx]
            all_sampled_indices = np.vstack([pos_sampled, neg_coords])

        # Randomize before the meta split so support and query both contain
        # positive and negative samples instead of one class each.
        permutation = np.random.permutation(len(all_sampled_indices))
        all_sampled_indices = all_sampled_indices[permutation]

        # Convert sampled voxel indices to physical coordinates (mm) using the affine,
        # then center the volume and scale so 1 INR unit = 10 cm.
        i, j, k = all_sampled_indices.T
        voxel_indices = np.stack([i, j, k, np.ones_like(i)], axis=1)
        physical_coords = (affine @ voxel_indices.T).T[:, :3]

        volume_center_voxel = (np.array(labels.shape) - 1) / 2.0
        volume_center_phys = affine[:3, :3] @ volume_center_voxel + affine[:3, 3]

        coords = np.concatenate([
            ((physical_coords - volume_center_phys) / 100.0),
            np.full((len(i), 1), time_point, dtype=np.float32)
        ], axis=1)

        labels_sampled = labels[i, j, k]

        return coords.astype(np.float32), labels_sampled.astype(np.float32)


class Validation_Extrapolation_LesionDataset(LesionDataset):
    def __init__(self, 
                 trajectories, 
                 device='cuda', 
                 dialation_iterations=5, 
                 background_samples=3000):
        super().__init__(trajectories, device=device, dialation_iterations=dialation_iterations, background_samples=background_samples)
        self.trajectories = [trj for trj in self.trajectories if f"{trj.patient_id}_{trj.label_id}" in self.validation_patients]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]

        dates_list = trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates
        extrapolation_timestep = dates_list[-1]

        coords, labels_sampled = self.prepare_data(extrapolation_timestep, trj)
        return coords, labels_sampled, self.get_embedding_id(trj)

class Validation_Interpolation_LesionDataset(LesionDataset):
    def __init__(self, 
                 trajectories, 
                 device='cuda', 
                 dialation_iterations=5, 
                 background_samples=3000):
        super().__init__(trajectories, device=device, dialation_iterations=dialation_iterations, background_samples=background_samples)
        self.trajectories = [trj for trj in self.trajectories if f"{trj.patient_id}_{trj.label_id}" in self.validation_patients]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]

        dates_list = trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates

        half_of_the_list = len(dates_list[-1]) // 2
        interpolation_timestep = dates_list[half_of_the_list]
        coords, labels_sampled = self.prepare_data(interpolation_timestep, trj)
        return coords, labels_sampled, self.get_embedding_id(trj)


class Plotting_LesionDataset(LesionDataset):
    def __init__(self, 
                 trajectories, 
                 device='cuda', 
                 dialation_iterations=5, 
                 background_samples=3000):
        super().__init__(trajectories, device=device, dialation_iterations=dialation_iterations, background_samples=background_samples)
        with open(DatasetPaths().plotting_samples(), "r") as f:
            self.plotting_samples = [sample["Patient"] + "_" + sample["Lesion"]  for sample in json.load(f)]

        self.trajectories = [trj for trj in self.trajectories if f"{trj.patient_id}_{trj.label_id}" in self.plotting_samples]

    def __getitem__(self, idx):
        trj = self.trajectories[idx]

        dates_list = trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates

        coords, labels = self.prepare_data(dates_list[0], trj)
        trj.embedding_id = self.get_embedding_id(trj)
        trj.coords = coords
        trj.labels = labels
        return self.trajectories[idx]

