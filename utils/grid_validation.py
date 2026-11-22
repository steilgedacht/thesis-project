"""Coarse-to-fine uniform-grid validation for continuous (INR / Neural ODE) models.

``validation_arena.py`` scores a model on a dense uniform grid over the cube
[-1, 1]^3 (model coordinates, 1 unit = 100 mm around the volume centre).
Doing that at full resolution every validation epoch is too expensive, so this
module approximates the same metric hierarchically:

1. The fine grid (``fine_resolution`` points per axis) is split into
   ``coarse_resolution``^3 sectors of ``(fine / coarse)``^3 fine points each.
2. The model is queried once per sector, at the sector centre.
3. A sector is *refined* if the coarse query predicted positive OR the ground
   truth has any positive fine point inside it; that set is then dilated by
   ``neighbor_sectors`` sectors in every direction.
4. Every fine point of a refined sector is queried. All other fine points are
   treated as predicted negative; by construction their ground truth is
   negative too, so they contribute nothing to Dice.

The only approximation is therefore false positives that live in a sector
whose centre and neighbouring sectors were all predicted negative and that
contain no lesion. Ground-truth positives are always evaluated exactly.

Coordinates, nearest-neighbour label sampling, Dice and the grid-point volume
are identical to ``validation_arena.py`` at ``--resolution fine_resolution``.
The fine ground truth of each (lesion, date) is computed once and cached
bit-packed (~0.4 MB per scan at 150^3), so later epochs only cost model calls.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import binary_dilation, map_coordinates


@dataclass
class _GridTarget:
    time_point: float
    packed_target: np.ndarray   # np.packbits of the flat (C-order) fine-grid mask
    target_sectors: np.ndarray  # bool [C, C, C]: sector contains a GT positive
    target_count: int


@dataclass
class GridCaseResult:
    dice_score: float
    predicted_count: int
    target_count: int
    intersection_count: int
    coarse_positive_sectors: int
    refined_sectors: int
    evaluated_points: int


class HierarchicalGridEvaluator:
    def __init__(
        self,
        coarse_resolution=15,
        fine_resolution=150,
        neighbor_sectors=1,
        chunk_size=262144,
        threshold=0.5,
        device="cuda",
    ):
        if fine_resolution % coarse_resolution != 0:
            raise ValueError(
                "fine_resolution must be a multiple of coarse_resolution "
                f"(got {fine_resolution} and {coarse_resolution})."
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        self.coarse = int(coarse_resolution)
        self.fine = int(fine_resolution)
        self.block = self.fine // self.coarse
        self.neighbor_sectors = int(neighbor_sectors)
        self.chunk_size = int(chunk_size)
        self.threshold = float(threshold)
        self.device = device

        # Same axis as validation_arena._coordinates_and_labels.
        self.fine_axis = (
            -1.0 + np.arange(self.fine, dtype=np.float64) * (2.0 / (self.fine - 1))
        ).astype(np.float32)
        # Sector centre = mean of the fine coordinates inside the sector.
        self.coarse_axis = self.fine_axis.reshape(self.coarse, self.block).mean(axis=1)
        self.grid_point_volume_cm3 = (200.0 / (self.fine - 1)) ** 3 / 1000.0

        offsets = np.arange(self.block)
        self._block_offsets = np.stack(
            np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self._targets = {}

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    def _target(self, trj, date):
        key = (trj.patient_id, trj.label_id, str(date))
        cached = self._targets.get(key)
        if cached is not None:
            return cached

        (labels, time_point), affine = trj.load_labels_for_inr(
            selected_date=str(date), absolute_day_number=True, affine=True
        )
        labels = np.asarray(labels, dtype=np.float32)
        voxel_center = (np.asarray(labels.shape, dtype=np.float64) - 1.0) / 2.0
        center_phys = affine[:3, :3] @ voxel_center + affine[:3, 3]
        inverse_affine = np.linalg.inv(affine)

        # Sample one slab of `block` planes along axis 0 at a time
        # (block * fine^2 points), i.e. one row of sectors.
        axis = self.fine_axis.astype(np.float64)
        grid_1, grid_2 = np.meshgrid(axis, axis, indexing="ij")
        plane = np.stack((grid_1.ravel(), grid_2.ravel()), axis=1)
        target = np.empty((self.fine, self.fine, self.fine), dtype=bool)
        for start in range(0, self.fine, self.block):
            slab_axis = axis[start:start + self.block]
            xyz = np.concatenate(
                (
                    np.repeat(slab_axis, len(plane))[:, None],
                    np.tile(plane, (len(slab_axis), 1)),
                ),
                axis=1,
            )
            physical = center_phys + xyz * 100.0
            voxel = inverse_affine[:3, :3] @ physical.T + inverse_affine[:3, 3:4]
            target[start:start + self.block] = (
                map_coordinates(labels, voxel, order=0, mode="constant", cval=0.0)
                > 0.5
            ).reshape(len(slab_axis), self.fine, self.fine)

        target_sectors = target.reshape(
            self.coarse, self.block, self.coarse, self.block, self.coarse, self.block
        ).any(axis=(1, 3, 5))
        cached = _GridTarget(
            time_point=float(time_point),
            packed_target=np.packbits(target.ravel()),
            target_sectors=target_sectors,
            target_count=int(target.sum()),
        )
        self._targets[key] = cached
        return cached

    # ------------------------------------------------------------------
    # Model queries
    # ------------------------------------------------------------------
    def _predict(self, query, xyz, time_point):
        """Boolean predictions for xyz [N, 3] (float32 numpy) at one time point."""
        predictions = np.empty(len(xyz), dtype=bool)
        for start in range(0, len(xyz), self.chunk_size):
            end = min(start + self.chunk_size, len(xyz))
            logits = query(xyz[start:end], time_point).reshape(-1)
            predictions[start:end] = (
                (torch.sigmoid(logits) >= self.threshold).cpu().numpy()
            )
        return predictions

    def coords_tensor(self, xyz, time_point):
        """[1, N, 4] model input (xyz + time) on the evaluator's device."""
        coords = torch.empty((1, len(xyz), 4), dtype=torch.float32)
        coords[0, :, :3] = torch.from_numpy(xyz)
        coords[0, :, 3] = time_point
        return coords.to(self.device)

    def target_time(self, trj, date):
        """Time point (absolute days) of the scan, as used for the grid queries."""
        return self._target(trj, date).time_point

    def evaluate(self, model, trj, date, embedding_id):
        """Evaluate a continuous model called as ``model(coords [1,N,4], patient_idx [1])``."""
        patient_tensor = torch.tensor(
            [int(embedding_id)], dtype=torch.long, device=self.device
        )
        return self.evaluate_query(
            lambda xyz, t: model(self.coords_tensor(xyz, t), patient_tensor),
            trj,
            date,
        )

    @torch.inference_mode()
    def evaluate_query(self, query, trj, date):
        """Evaluate ``query(xyz [N,3] float32 numpy, time_point) -> logits`` (N values)."""
        target = self._target(trj, date)

        # 1) coarse pass: one query per sector centre
        c = self.coarse_axis
        coarse_xyz = np.stack(
            np.meshgrid(c, c, c, indexing="ij"), axis=-1
        ).reshape(-1, 3).astype(np.float32)
        coarse_positive = self._predict(
            query, coarse_xyz, target.time_point
        ).reshape(self.coarse, self.coarse, self.coarse)

        # 2) sectors to refine: predicted or GT positive, plus their neighbours
        refine = coarse_positive | target.target_sectors
        if self.neighbor_sectors > 0 and refine.any():
            refine = binary_dilation(
                refine,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=self.neighbor_sectors,
            )

        # 3) fine pass over all points of the refined sectors
        sectors = np.argwhere(refine)
        if len(sectors) == 0:
            predicted_count = intersection_count = evaluated_points = 0
        else:
            fine_index = (
                sectors[:, None, :] * self.block + self._block_offsets[None, :, :]
            ).reshape(-1, 3)
            evaluated_points = len(fine_index)
            fine_xyz = self.fine_axis[fine_index]
            predictions = self._predict(query, fine_xyz, target.time_point)
            flat = (
                fine_index[:, 0] * self.fine * self.fine
                + fine_index[:, 1] * self.fine
                + fine_index[:, 2]
            )
            targets = (
                (target.packed_target[flat >> 3] >> (7 - (flat & 7))) & 1
            ).astype(bool)
            predicted_count = int(predictions.sum())
            intersection_count = int(np.logical_and(predictions, targets).sum())

        dice = (2.0 * intersection_count + 1e-6) / (
            predicted_count + target.target_count + 1e-6
        )
        return GridCaseResult(
            dice_score=float(dice),
            predicted_count=predicted_count,
            target_count=target.target_count,
            intersection_count=intersection_count,
            coarse_positive_sectors=int(coarse_positive.sum()),
            refined_sectors=int(len(sectors)),
            evaluated_points=int(evaluated_points),
        )

    def wrongly_predicted_volume_cm3(self, result):
        return abs(result.predicted_count - result.target_count) * self.grid_point_volume_cm3


def evaluator_from_config(config):
    """Build an evaluator from the ``validation_grid_*`` config attributes."""
    return HierarchicalGridEvaluator(
        coarse_resolution=getattr(config, "validation_grid_coarse_resolution", 15),
        fine_resolution=getattr(config, "validation_grid_fine_resolution", 150),
        neighbor_sectors=getattr(config, "validation_grid_neighbor_sectors", 1),
        chunk_size=getattr(config, "validation_grid_chunk_size", 262144),
        threshold=getattr(config, "validation_prediction_threshold", 0.5),
        device=config.device,
    )


def lesion_validation_cases(dataset):
    """Yield (trajectory, target date, embedding id, days T) for the
    Validation_{Extrapolation,Interpolation}_LesionDataset family (utils/dataloader.py)."""
    for item in dataset.trajectories:
        if isinstance(item, tuple) and len(item) == 3:  # extrapolation: (T, date, trj)
            T, date, trj = item
        else:  # interpolation: the trajectory itself
            trj, T = item, -1
            dates = trj.allowed_dates if hasattr(trj, "allowed_dates") else trj.dates
            lesion_id = f"{trj.patient_id}_{trj.label_id}"
            date = dates[int(dataset.validation_samples[lesion_id]["interpolation"])]
        yield trj, str(date), dataset.get_embedding_id(trj), float(T)


def grid_validation_metrics(prefix, lesion_ids, results, evaluator, extrapolation):
    """MLflow metrics for one split, aggregated like validation_arena._summary_metrics.

    Returns (metrics dict, mean Dice).
    """
    if not results:
        raise ValueError(f"{prefix}: no validation samples.")
    dice_scores = [result.dice_score for result in results]
    volume_errors = [evaluator.wrongly_predicted_volume_cm3(result) for result in results]
    if extrapolation:
        mean_dice = extrapolation_weighted_mean(lesion_ids, dice_scores)
        average_volume_error = extrapolation_weighted_mean(lesion_ids, volume_errors)
    else:
        mean_dice = float(np.mean(dice_scores))
        average_volume_error = float(np.mean(volume_errors))
    n_cases = len(results)
    metrics = {
        f"{prefix}_dice_loss": 1.0 - mean_dice,
        f"{prefix}_dice_score": mean_dice,
        f"{prefix}_median_wrongly_predicted_volume_cm3": float(np.median(volume_errors)),
        f"{prefix}_average_wrongly_predicted_volume_cm3": average_volume_error,
        f"{prefix}_wrongly_predicted_volume_cm3": float(np.sum(volume_errors)),
        # cost / coverage diagnostics of the coarse-to-fine search
        f"{prefix}_grid_refined_sectors_per_case":
            sum(r.refined_sectors for r in results) / n_cases,
        f"{prefix}_grid_coarse_positive_sectors_per_case":
            sum(r.coarse_positive_sectors for r in results) / n_cases,
        f"{prefix}_grid_evaluated_point_fraction":
            sum(r.evaluated_points for r in results) / (n_cases * evaluator.fine ** 3),
    }
    return metrics, mean_dice


def extrapolation_weighted_mean(lesion_ids, values):
    """Aggregate like validation_arena._average_metric for the extrapolation split.

    Values are averaged per lesion first; each lesion is then weighted by its
    number of extrapolation timepoints divided by the maximum count.
    """
    by_lesion = {}
    for lesion_id, value in zip(lesion_ids, values):
        by_lesion.setdefault(lesion_id, []).append(float(value))
    max_count = max(len(v) for v in by_lesion.values())
    means = [float(np.mean(v)) for v in by_lesion.values()]
    weights = [len(v) / max_count for v in by_lesion.values()]
    return float(np.average(means, weights=weights))
