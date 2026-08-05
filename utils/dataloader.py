from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import binary_dilation

class LesionDataset(Dataset):
    def __init__(self, 
                 trajectories, 
                 device='cuda', 
                 dialation_iterations=5, 
                 mode='train', 
                 val_date_idx=None, 
                 val_end_date_idx=None, 
                 background_samples=3000):
        self.device = device
        self.trajectories = trajectories

        self.dialation_iterations = dialation_iterations
        self.patient_to_idx = {p.patient_id: i for i, p in enumerate(trajectories)}
        self.idx_to_patient = {i: p.patient_id for i, p in enumerate(trajectories)}
        self.mode = mode
        self.max_samples = background_samples
        self.proportion_dialation_background = 5
        self.proportion_pos_neg_samples = 2

        if val_date_idx is None:
            self.val_date_idx = [
                np.random.choice(trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates[1:-1]) 
                for trj in trajectories
            ]
        else:
            self.val_date_idx = val_date_idx

        if val_end_date_idx is None:
            self.val_end_date_idx = np.random.choice(range(len(trajectories)), size=len(trajectories) // 10, replace=False)
        else:
            self.val_end_date_idx = val_end_date_idx
        
        if self.mode == 'valid_extrapolation':
            self.trajectories = [trj for i, trj in enumerate(trajectories) if i in self.val_end_date_idx]

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        patient_id = self.patient_to_idx[trj.patient_id]

        if self.mode == 'train':
            dates_list = trj.dates
            if idx in self.val_end_date_idx:
                dates_list = trj.dates[:-1]
            if hasattr(trj, 'allowed_dates'):
                dates_list = trj.allowed_dates
            random_time_point = str(np.random.choice(dates_list))
        else:
            if self.mode == 'valid_extrapolation':
                random_time_point = str(trj.allowed_dates[-1] if hasattr(trj, 'allowed_dates') else trj.dates[-1])
            else: # self.mode == 'valid_interpolation':
                random_time_point = str(self.val_date_idx[idx])

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

        return coords.astype(np.float32), labels_sampled.astype(np.float32), np.int64((patient_id * 100) + self.trajectories[idx].label_id)

