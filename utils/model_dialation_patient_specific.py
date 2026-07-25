import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import binary_dilation
import functools



class Patient_Specific_Dilation_Model(nn.Module):
    def __init__(self, trajectories, shape=(500, 500, 50)):
        super().__init__()
        self.shape = shape
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        self.trajectories_map = {}
        self.rates_per_lesion = {} # Speichert k_i für jede Läsion
        self.patient_to_idx = {p.patient_id: i for i, p in enumerate(trajectories)}

        print("Pre-calculating growth rates per lesion trajectory...")
        for trj in trajectories:
            label_key = (self.patient_to_idx[trj.patient_id] * 100) + trj.label_id
            self.trajectories_map[label_key] = trj
            
            t_first_str = str(trj.dates[0] if not hasattr(trj, 'allowed_dates') else trj.allowed_dates[0])
            t_last_str = str(trj.dates[-1] if not hasattr(trj, 'allowed_dates') else trj.allowed_dates[-1])
            
            mask_first, t0 = trj.load_labels_for_inr(selected_date=t_first_str, absolute_day_number=True)
            mask_last, tT = trj.load_labels_for_inr(selected_date=t_last_str, absolute_day_number=True)
            
            dt = max(1.0, tT - t0)
            
            # approximate the lesion as a sphere to calculate the radius
            vol_first = np.sum(mask_first > 0.5)
            vol_last = np.sum(mask_last > 0.5)
            
            r_first = (3 * vol_first / (4 * np.pi)) ** (1/3)
            r_last = (3 * vol_last / (4 * np.pi)) ** (1/3)
            
            # pixel growth rate per day
            growth_rate = max(0.0, (r_last - r_first) / dt)
            self.rates_per_lesion[label_key] = growth_rate

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
            dilation_iterations = int(np.round(dt * self.rates_per_lesion[p_id]))
            
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