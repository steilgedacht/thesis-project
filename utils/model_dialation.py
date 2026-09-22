import json
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import binary_dilation
import functools

class Dilation_Model(nn.Module):
    def __init__(self, trajectories, shape=(500, 500, 50), pixels_per_day=0.0026281461313099595):
        super().__init__()
        self.shape = shape
        self.pixels_per_day = pixels_per_day
        self.dummy_param = nn.Parameter(torch.zeros(1)) # for the optimizer to have parameters to optimize

        self.trajectories_map = {}
        with open("experiments/05_data_visualizer/patient_to_idx.json", "r") as f:
            self.patient_to_idx = json.load(f)  

        for trj in trajectories:
            label_key = (self.patient_to_idx[trj.patient_id] * 100) + trj.label_id
            self.trajectories_map[label_key] = trj

    @functools.lru_cache(maxsize=16) 
    def _load_base_scan(self, p_id):
        trj = self.trajectories_map[p_id]
        first_date = str(trj.allowed_dates[0])
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
        # The inverse is physical = coords * 100 + volume_center_phys, then
        # voxel = affine^-1 @ [physical, 1]. This must match the code in
        # LesionDataset.prepare_data exactly.
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
            dilation_iterations = int(np.round(dt * self.pixels_per_day))
            current_mask_np = binary_dilation(base_mask_np, iterations=dilation_iterations) if dilation_iterations > 0 else base_mask_np
            current_mask = torch.from_numpy(current_mask_np.astype(np.uint8)).to(device)

            sampled_values = current_mask[pixel_coords[:, 0], pixel_coords[:, 1], pixel_coords[:, 2]]
            logits = torch.where(sampled_values, 15.0, -15.0)
            out[b, :, 0] = logits

        return out