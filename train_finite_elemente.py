import os
import gc
import mlflow
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from utils.mri_dataloader import MRI_Dataloader
from utils.model_fem_solver import FEMLesionGrowthSolver, crop_and_resample_trajectory

# Fixed mesh / data resolution used for every trajectory. Because every
# trajectory is cropped and resampled to this exact shape before it hits
# the solver, the FEM mesh and the data grid always match - no more
# up-interpolation from a coarse solve to full native MRI resolution.
CROP_SHAPE = (32, 32, 32)

# Padding around the t0 lesion bbox, as a fraction of its own extent.
# Needs enough margin to contain growth out to your longest extrapolation
# horizon (t_extrap), not just t0 -> t1 - tune against your data.
PADDING_FRAC = 0.6

# Voxel spacing is derived automatically per-trajectory from each scan's
# NIfTI affine matrix inside crop_and_resample_trajectory() - no need to
# hardcode it here.


def compute_dice_score(pred_mask, target_mask, threshold=0.5):
    pred_binary = (pred_mask > threshold).astype(np.float32)
    target_binary = (target_mask > threshold).astype(np.float32)
    intersection = np.sum(pred_binary * target_binary)
    return (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-8)


def fit_fem_for_trajectory(cropped, solver):
    """
    Fits D and rho on the cropped/resampled trajectory (t0 -> t1).
    `cropped` is the dict returned by crop_and_resample_trajectory().
    """
    times = cropped["times"]
    masks = cropped["masks"]

    t_0, mask_t0 = times[0], masks[0]
    t_1, mask_t1 = times[1], masks[1]

    dt_days = t_1 - t_0
    num_steps = 10
    dt = dt_days / num_steps

    def objective(params):
        D, rho = params
        if D < 0 or rho < 0:
            return 1e6  # strict penalty for unphysical values

        pred_t1 = solver.simulate(mask_t0, D_val=D, rho_val=rho, dt=dt, num_steps=num_steps)
        return 1.0 - compute_dice_score(pred_t1, mask_t1)

    initial_params = [0.01, 0.05]
    res = minimize(objective, initial_params, method='Nelder-Mead', options={'maxiter': 30})

    best_D, best_rho = res.x
    return best_D, best_rho, res.fun


def run_fem_baseline():
    mlflow.set_experiment("FEM_Lesion_Baseline")

    with mlflow.start_run(run_name="FEM_Optimization"):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(
            n_scans=3,  # at least 3 scans for train (t0->t1), interp, extrap
            only_growing=True,
        )
        trajectories = mri_dataloader.cache_lesion_trajectories
        assert trajectories is not None, "Keine Trajektorien geladen!"

        print(f"Starte FEM-Fitting für {len(trajectories)} Trajektorien...")

        interp_dice_scores = []
        extrap_dice_scores = []

        for p_idx, trj in enumerate(tqdm(trajectories, desc="Fitting FE-Modelle")):
            # crop around the t0 lesion, pad, resample every timepoint onto
            # a fixed 32^3 grid. The crop is sized from t0 ONLY, so nothing
            # from the future (t_interp / t_extrap) leaks into it. Voxel
            # spacing is derived automatically from the NIfTI affine.
            try:
                cropped = crop_and_resample_trajectory(
                    trj,
                    target_shape=CROP_SHAPE,
                    padding_frac=PADDING_FRAC,
                )
            except ValueError as e:
                print(f"Skipping trajectory {p_idx}: {e}")
                continue

            solver = FEMLesionGrowthSolver(
                target_grid_shape=CROP_SHAPE,
                domain_bounds=cropped["domain_bounds"],
            )

            # 1. fit D and rho on t0 -> t1
            best_D, best_rho, train_loss = fit_fem_for_trajectory(cropped, solver)

            times = cropped["times"]
            masks = cropped["masks"]
            t_0, mask_t0 = times[0], masks[0]

            # 2. evaluation: interpolation
            t_interp, mask_interp_gt = times[1], masks[1]
            dt = (t_interp - t_0) / 10
            pred_interp = solver.simulate(mask_t0, best_D, best_rho, dt=dt, num_steps=10)
            interp_dice = compute_dice_score(pred_interp, mask_interp_gt)
            interp_dice_scores.append(interp_dice)

            # 3. evaluation: extrapolation (future)
            t_extrap, mask_extrap_gt = times[-1], masks[-1]
            dt_extrap = (t_extrap - t_0) / 20
            pred_extrap = solver.simulate(mask_t0, best_D, best_rho, dt=dt_extrap, num_steps=20)
            extrap_dice = compute_dice_score(pred_extrap, mask_extrap_gt)
            extrap_dice_scores.append(extrap_dice)

            # release PETSc objects for this trajectory before starting the
            # next one, instead of leaving it to the garbage collector
            solver.close()
            del solver
            gc.collect()

        mean_interp_dice = np.mean(interp_dice_scores) if interp_dice_scores else 0.0
        mean_extrap_dice = np.mean(extrap_dice_scores) if extrap_dice_scores else 0.0

        mlflow.log_metrics({
            "valid_interpolation_dice_score": mean_interp_dice,
            "valid_interpolation_dice_loss": 1.0 - mean_interp_dice,
            "valid_extrapolation_dice_score": mean_extrap_dice,
            "valid_extrapolation_dice_loss": 1.0 - mean_extrap_dice,
        })

        print(f"\n--- Ergebnisse ---")
        print(f"Interpolation Dice Score:  {mean_interp_dice:.4f}")
        print(f"Extrapolation Dice Score: {mean_extrap_dice:.4f}")


if __name__ == "__main__":
    run_fem_baseline()