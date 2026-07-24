import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import binary_dilation
import functools

class Dilation_Model(nn.Module):
    def __init__(self, trajectories, shape=(500, 500, 50), pixels_per_day=150/(365*5)):
        super().__init__()
        self.shape = shape
        self.pixels_per_day = pixels_per_day
        self.dummy_param = nn.Parameter(torch.zeros(1)) # for the optimizer to have parameters to optimize

        self.trajectories_map = {}
        self.patient_to_idx = {p.patient_id: i for i, p in enumerate(trajectories)}

        for trj in trajectories:
            label_key = (self.patient_to_idx[trj.patient_id] * 100) + trj.label_id
            self.trajectories_map[label_key] = trj

    @functools.lru_cache(maxsize=16) 
    def _load_base_scan(self, p_id):
        trj = self.trajectories_map[p_id]
        first_date = str(trj.dates[0])
        labels, __annotations__ = trj.load_labels_for_inr(selected_date=first_date, absolute_day_number=True)        
        mask_bool = labels > 0.5 
        return mask_bool

    def forward(self, x, patient_idx):
        """
        x: Shape [Batch, Num_Samples, 4] -> (x, y, z, t)
        patient_idx: Shape [Batch]
        """
        device = x.device
        batch_size, num_samples, _ = x.shape
        out = torch.zeros(batch_size, num_samples, 1, device=device)

        # scale coordinates from [-1, 1] to [0, shape-1]
        coords_xyz = x[..., :3]
        shape_tensor = torch.tensor(self.shape, device=device).view(1, 1, 3)
        pixel_coords = ((coords_xyz + 1.0) / 2.0) * (shape_tensor - 1.0)
        pixel_coords = torch.round(pixel_coords).long()

        # Loop over batch dimension to handle each patient separately
        for b in range(batch_size):
            p_id = patient_idx[b].item()
            
            base_mask_np = self._load_base_scan(p_id)
            
            dt = x[b, 0, 3].item() 
            dilation_iterations = int(np.round(dt * self.pixels_per_day))
            
            if dilation_iterations > 0:
                current_mask_np = binary_dilation(base_mask_np, iterations=dilation_iterations)
            else:
                current_mask_np = base_mask_np
            
            current_mask = torch.from_numpy(current_mask_np).to(device)
            
            idx_x = pixel_coords[b, :, 0]
            idx_y = pixel_coords[b, :, 1]
            idx_z = pixel_coords[b, :, 2]
            
            sampled_values = current_mask[idx_x, idx_y, idx_z]
            
            # Convert to logits for BCEWithLogitsLoss
            logits = torch.where(sampled_values, 15.0, -15.0)
            out[b, :, 0] = logits
            
        return out