import json
import torch
import functools
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage import binary_dilation


class Patient_Specific_Dilation_Model(nn.Module):
    def __init__(self, trajectories, shape=(500, 500, 50)):
        super().__init__()
        self.shape = shape
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
        self.trajectories_map = {}
        self.rates_per_lesion = {} # Speichert k_i für jede Läsion
        with open("experiments/05_data_visualizer/patient_to_idx.json", "r") as f:
            self.patient_to_idx = json.load(f)  

        print("Pre-calculating growth rates per lesion trajectory...")
        for trj in tqdm(list(set(trajectories)), total=len(set(trajectories))):
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
        first_date = str(trj.allowed_dates[0] if hasattr(trj, 'allowed_dates') else trj.dates[0])
        packed, affine = trj.load_labels_for_inr(selected_date=first_date, absolute_day_number=True, affine=True)
        labels = packed[0] if isinstance(packed, tuple) else packed
        mask_bool = labels > 0.5
        return mask_bool, affine

    def forward(self, x, patient_idx):
        """
        x: Shape [Batch, Num_Samples, 4] -> (x, y, z, t)
        patient_idx: Shape [Batch]
        """
        device = x.device
        batch_size, num_samples, _ = x.shape
        out = torch.zeros(batch_size, num_samples, 1, device=device)

        # The training data uses a physical-space normalization with the volume
        # center as origin: coords = (physical - volume_center_phys) / 100.0.
        # We therefore need the exact inverse affine transform back to voxel
        # coordinates before dilating/sampling the base mask.
        coords_xyz = x[..., :3]

        for b in range(batch_size):
            p_id = patient_idx[b].item()
            base_mask_np, affine = self._load_base_scan(p_id)

            volume_center_voxel = (np.asarray(base_mask_np.shape) - 1) / 2.0
            volume_center_phys = affine[:3, :3] @ volume_center_voxel + affine[:3, 3]

            physical_coords = coords_xyz[b] * 100.0 + torch.tensor(volume_center_phys, device=device, dtype=coords_xyz.dtype)
            ones = torch.ones((num_samples, 1), device=device, dtype=coords_xyz.dtype)
            physical_hom = torch.cat([physical_coords, ones], dim=-1)

            affine_inv_t = torch.tensor(np.linalg.inv(affine), device=device, dtype=coords_xyz.dtype)
            voxel_hom = torch.matmul(physical_hom, affine_inv_t.T)
            pixel_coords = torch.round(voxel_hom[:, :3]).long()

            max_idx = torch.tensor(np.asarray(base_mask_np.shape) - 1, device=device, dtype=torch.long)
            pixel_coords[:, 0] = pixel_coords[:, 0].clamp(0, max_idx[0])
            pixel_coords[:, 1] = pixel_coords[:, 1].clamp(0, max_idx[1])
            pixel_coords[:, 2] = pixel_coords[:, 2].clamp(0, max_idx[2])

            dt = x[b, 0, 3].item()
            dilation_iterations = int(np.round(dt * self.rates_per_lesion[p_id]))
            current_mask_np = binary_dilation(base_mask_np, iterations=dilation_iterations) if dilation_iterations > 0 else base_mask_np
            current_mask = torch.from_numpy(current_mask_np.astype(np.uint8)).to(device)

            sampled_values = current_mask[pixel_coords[:, 0], pixel_coords[:, 1], pixel_coords[:, 2]]
            logits = torch.where(sampled_values, 15.0, -15.0)
            out[b, :, 0] = logits

        return out