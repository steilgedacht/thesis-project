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
        self.shape = (500, 500, 50)
        self.meshgrid = np.meshgrid(np.linspace(0, 1, self.shape[0]),
                              np.linspace(0, 1, self.shape[1]),
                              np.linspace(0, 1, self.shape[2]),
                              indexing='ij')
        self.dialation_iterations = dialation_iterations
        self.patient_to_idx = {p.patient_id: i for i, p in enumerate(trajectories)}
        self.idx_to_patient = {i: p.patient_id for i, p in enumerate(trajectories)}
        self.mode = mode
        self.max_samples = background_samples

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
        patient_idx = self.patient_to_idx[trj.patient_id]

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

        labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point, absolute_day_number=True)
        
        lesion_mask = labels > 0.5        
        pos_coords = np.argwhere(lesion_mask)

        border_samples = binary_dilation(labels, iterations=self.dialation_iterations) - labels
        border_samples_coords = np.argwhere(border_samples)

        num_neg_needed = self.max_samples // 2
        neg_coords = border_samples_coords.tolist()

        if len(neg_coords) > num_neg_needed // 2:
            neg_coords = [neg_coords[i] for i in np.random.choice(len(neg_coords), size=num_neg_needed // 2, replace=True).tolist()]

        while len(neg_coords) < num_neg_needed:
            candidate_coords = np.array([
                np.random.randint(0, s, num_neg_needed) for s in self.shape
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
        
        i, j, k = all_sampled_indices.T
        coords = np.stack([
            self.meshgrid[0][i, j, k] * 2 - 1,
            self.meshgrid[1][i, j, k] * 2 - 1,
            self.meshgrid[2][i, j, k] * 2 - 1,
            np.full(len(i), time_point)
        ], axis=1)
        
        labels_sampled = labels[i, j, k]

        return coords.astype(np.float32), labels_sampled.astype(np.float32), np.int64((patient_idx * 100) + self.trajectories[idx].label_id)
