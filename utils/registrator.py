"""Rigid-registration helper. No dependency on any other class in the package."""

import torch
from scipy.ndimage import affine_transform


class Registrator:
    def __init__(self, device=None):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def get_3d_bbox(self, mask):
        z_idx = torch.nonzero(mask.sum(dim=(1, 2)))
        y_idx = torch.nonzero(mask.sum(dim=(0, 2)))
        x_idx = torch.nonzero(mask.sum(dim=(0, 1)))

        bbox = {
            "z_min": z_idx[0].item(), "z_max": z_idx[-1].item(),
            "y_min": y_idx[0].item(), "y_max": y_idx[-1].item(),
            "x_min": x_idx[0].item(), "x_max": x_idx[-1].item(),
        }

        center = torch.tensor([
            (bbox["x_min"] + bbox["x_max"]) / 2.0,
            (bbox["y_min"] + bbox["y_max"]) / 2.0,
            (bbox["z_min"] + bbox["z_max"]) / 2.0,
        ], device=self.device)

        return bbox, center

    def rigid_transform(self, moving_image, params, output_shape):
        tx, ty, tz, rx, ry, rz, sx, sy, sz = params

        transformed = affine_transform(
            moving_image,
            [sx, sy, sz],
            offset=[tx, ty, tz],
            output_shape=output_shape,
            order=1,
            mode="constant",
            cval=0
        )
        return transformed

    def register(self, reference_mask, moving_mask):
        ref_t = torch.from_numpy(reference_mask).float().permute(2, 1, 0).to(self.device)
        mov_t = torch.from_numpy(moving_mask).float().permute(2, 1, 0).to(self.device)

        bbox_ref, center_ref = self.get_3d_bbox(ref_t)
        bbox_mov, center_mov = self.get_3d_bbox(mov_t)

        sx = (bbox_mov["x_max"] - bbox_mov["x_min"]) / (bbox_ref["x_max"] - bbox_ref["x_min"])
        sy = (bbox_mov["y_max"] - bbox_mov["y_min"]) / (bbox_ref["y_max"] - bbox_ref["y_min"])
        sz = (bbox_mov["z_max"] - bbox_mov["z_min"]) / (bbox_ref["z_max"] - bbox_ref["z_min"])

        tx = center_mov[0] - center_ref[0] * sx
        ty = center_mov[1] - center_ref[1] * sy
        tz = center_mov[2] - center_ref[2] * sz

        init_params = torch.tensor([
            tx, ty, tz,
            0.0, 0.0, 0.0,
            sx, sy, sz
        ])

        return init_params.cpu().detach().numpy()