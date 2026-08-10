import json
import torch
import numpy as np
import torch.nn as nn
from datetime import datetime
from .paths import DatasetPaths
from scipy.ndimage import distance_transform_edt
 

class LinearInterpolationBaseline(nn.Module):
    def __init__(self, trajectories):
        super().__init__()
        self.trajectories = trajectories
        self.dummy_param = nn.Parameter(torch.zeros(1)) # for the optimizer to have parameters to optimize

        with open(DatasetPaths().patient_to_idx(), "r") as f:
            self.patient_to_idx = json.load(f)

        self.patient_to_trajectory = {(self.patient_to_idx[trj.patient_id] * 100) + trj.label_id : trj for trj in trajectories}

    def distance_field_interpolation(self, shape_a, shape_b, t):
        mask_a = shape_a > 0.5
        mask_b = shape_b > 0.5

        dt_a = distance_transform_edt(mask_a) - distance_transform_edt(~mask_a)
        dt_b = distance_transform_edt(mask_b) - distance_transform_edt(~mask_b)

        dt_t = (1.0 - t) * dt_a + t * dt_b

        return (dt_t > 0).astype(np.uint8)

    def find_time_points(self, trj, dates, array, value):
        time_point_before = time_point_after = None
        for i, a in enumerate(array):
            if a < value: continue
            time_point_before = trj.allowed_dates[max(0, i-1)] 
            time_point_after  = trj.allowed_dates[min(len(dates) - 1, i)]
            break 
        if time_point_before == None:
            time_point_before = time_point_after = trj.allowed_dates[-1]
        return time_point_before, time_point_after

    def predict(self, x, patient_idx):
        trj = self.patient_to_trajectory[patient_idx.item()]

        time_point = x[0, -1].item()
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in trj.allowed_dates]
        first_date = dates[0]
        days_since_first = [((d - first_date).days) for d in dates]

        # Safely find indices avoiding out-of-bounds wrapping
        time_point_before, time_point_after = self.find_time_points(trj, dates, days_since_first, time_point)
        
        (labels_before, time_point_before_), affine = trj.load_labels_for_inr(selected_date=time_point_before, absolute_day_number=True, affine=True)
        (labels_after, time_point_after_), _ = trj.load_labels_for_inr(selected_date=time_point_after, absolute_day_number=True, affine=True)

        t = (time_point - time_point_before_) / (time_point_after_ - time_point_before_ + 1e-6)

        interpolated = torch.tensor(self.distance_field_interpolation(labels_before, labels_after, t), device="cuda")

        # --- REVERSE DILATION / TRANSFORMATION PIPELINE ---
        coords_xyz = x[..., :3] # Shape: (1, N, 3)
        
        # 1. Unscale from INR units back to physical millimeters (mm) (reverses / 100.0)
        physical_coords = coords_xyz * 100.0

        # 2. Recompute volume center in physical space using the 'before' affine

        volume_center_voxel = (np.array(labels_before.shape) - 1) / 2.0
        volume_center_phys = affine[:3, :3] @ volume_center_voxel + affine[:3, 3]
        
        volume_center_phys_t = torch.tensor(volume_center_phys, device="cuda", dtype=coords_xyz.dtype).view(1, 1, 3)
        affine_inv_t = torch.tensor(np.linalg.inv(affine), device="cuda", dtype=coords_xyz.dtype)

        # 3. Add volume center offset back (reverses - volume_center_phys)
        physical_coords_centered = physical_coords + volume_center_phys_t

        # 4. Apply inverse affine to map physical (mm) back to voxel space (reverses affine @ voxel_indices.T)
        batch_size, num_points, _ = physical_coords_centered.shape
        ones = torch.ones((batch_size, num_points, 1), device="cuda", dtype=coords_xyz.dtype)
        physical_hom = torch.cat([physical_coords_centered, ones], dim=-1)

        voxel_hom = torch.matmul(physical_hom, affine_inv_t.T)
        pixel_coords = voxel_hom[..., :3]

        # 5. Round, convert to long, and extract indices (matching i, j, k order)
        pixel_coords = torch.round(pixel_coords).long()

        idx_x = pixel_coords[0, :, 0]
        idx_y = pixel_coords[0, :, 1]
        idx_z = pixel_coords[0, :, 2]
        
        # Clamp to prevent any floating-point edge-case out-of-bounds errors
        idx_x = torch.clamp(idx_x, 0, interpolated.shape[0] - 1)
        idx_y = torch.clamp(idx_y, 0, interpolated.shape[1] - 1)
        idx_z = torch.clamp(idx_z, 0, interpolated.shape[2] - 1)
        
        sampled_values = interpolated[idx_x, idx_y, idx_z]
        
        # Convert to logits for BCEWithLogitsLoss
        logits = torch.where(sampled_values, 15.0, -15.0)
        return logits

    def forward(self, x, patient_idx):
        outs = [self.predict(x[b], patient_idx) for b in range(x.shape[0])]
        return torch.stack(outs, dim=0).unsqueeze(-1)

