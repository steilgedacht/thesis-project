"""Region prior for the coordinate models (INR, Neural ODE).

Every lesion gets an axis-aligned box, in model coordinates, around the union
of its *training* scan masks (validation dates excluded, same rule as the
datasets). A query point that lies more than ``margin_mm`` outside that box
gets

    logit -= strength_per_mm * (distance_to_box_mm - margin_mm)

so the model cannot put lesion mass far from where the lesion was observed.
Within box + margin the network is unconstrained, and the box does not depend
on time, so the prediction stays continuous in t.

The boxes live in the model as a buffer, which means training, plotting, grid
validation and the validation arena all get the prior through the normal
``model(coords, patient_idx)`` call. Lesions without a box (NaN row) get no
penalty, so old behaviour is the fallback.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from .dataloader_point_lstm import all_dates, lesion_key, load_validation_samples, training_dates


class RegionPrior(nn.Module):
    def __init__(self, num_lesions, margin_mm=10.0, strength_per_mm=1.0):
        super().__init__()
        self.margin_mm = float(margin_mm)
        self.strength_per_mm = float(strength_per_mm)
        # [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z] in model coordinates, NaN = unknown lesion
        self.register_buffer("boxes", torch.full((int(num_lesions), 6), float("nan")))

    @torch.no_grad()
    def set_box(self, lesion_idx, box):
        self.boxes[int(lesion_idx)] = torch.as_tensor(
            np.asarray(box, dtype=np.float32), device=self.boxes.device
        )

    def get_box(self, lesion_idx):
        return self.boxes[int(lesion_idx)].detach().cpu().numpy().copy()

    def penalty(self, xyz, patient_idx):
        """xyz [B, N, 3], patient_idx [B] -> penalty [B, N, 1] (>= 0)."""
        box = self.boxes[patient_idx]                                  # [B, 6]
        valid = ~torch.isnan(box).any(dim=-1)                          # [B]
        # Replace NaN rows before any arithmetic so no NaN reaches the backward pass.
        box = torch.where(valid[:, None], box, torch.zeros_like(box))
        lo, hi = box[:, None, :3], box[:, None, 3:]
        outside = torch.clamp(lo - xyz, min=0) + torch.clamp(xyz - hi, min=0)
        distance_mm = torch.sqrt((outside ** 2).sum(dim=-1, keepdim=True) + 1e-12) * 100.0
        penalty = self.strength_per_mm * torch.relu(distance_mm - self.margin_mm)
        return penalty * valid[:, None, None].to(penalty.dtype)


def apply_region_prior(model, logits, xyz, patient_idx):
    """Subtract the region penalty if the model has a prior (old checkpoints have none)."""
    prior = getattr(model, "region_prior", None)
    if prior is None:
        return logits
    # Models built with region_prior_in_training=False only get the prior in
    # eval mode (validation, plotting, arena). Old checkpoints have no flag.
    if model.training and not getattr(model, "region_prior_in_training", True):
        return logits
    return logits - prior.penalty(xyz, patient_idx)


# ---------------------------------------------------------------------------
# Box computation
# ---------------------------------------------------------------------------
def mask_box(trajectory, dates):
    """Box around the union of the masks at ``dates`` in model coordinates, or None."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for date in dates:
        (labels, _), affine = trajectory.load_labels_for_inr(
            selected_date=str(date), absolute_day_number=True, affine=True
        )
        positive = np.argwhere(np.asarray(labels) > 0.5)
        if len(positive) == 0:
            continue
        affine = np.asarray(affine, dtype=np.float64)
        center_voxel = (np.asarray(np.shape(labels), dtype=np.float64) - 1.0) / 2.0
        center_phys = affine[:3, :3] @ center_voxel + affine[:3, 3]
        # Voxel corners, not centres, so the box covers the whole lesion.
        corners = np.concatenate([positive - 0.5, positive + 0.5]).astype(np.float64)
        xyz = (corners @ affine[:3, :3].T + affine[:3, 3] - center_phys) / 100.0
        lo = np.minimum(lo, xyz.min(axis=0))
        hi = np.maximum(hi, xyz.max(axis=0))
    if not np.isfinite(lo).all():
        return None
    return np.concatenate([lo, hi]).astype(np.float32)


def attach_region_boxes(model, trajectories, embedding_id, threads=8):
    """Fill ``model.region_prior`` with training-date boxes. Returns the number of boxes set.

    ``embedding_id(trajectory)`` must give the same index the datasets use.
    Does nothing (returns 0) if the model has no region prior.
    """
    prior = getattr(model, "region_prior", None)
    if prior is None:
        return 0
    validation_samples = load_validation_samples()
    unique = list({lesion_key(t): t for t in trajectories}.values())

    def job(trajectory):
        return int(embedding_id(trajectory)), mask_box(
            trajectory, training_dates(trajectory, validation_samples)
        )

    count = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for index, box in pool.map(job, unique):
            if box is None:
                continue
            if index >= prior.boxes.shape[0]:
                raise ValueError(f"Embedding id {index} does not fit the region prior table.")
            prior.set_box(index, box)
            count += 1
    print(f"Region prior: boxes for {count}/{len(unique)} lesions "
          f"(margin {prior.margin_mm:g} mm, {prior.strength_per_mm:g} logit/mm).")
    return count


def all_dates_box(trajectory):
    """Box over every scan of a lesion (used for the embedding test on a new lesion)."""
    return mask_box(trajectory, all_dates(trajectory))
