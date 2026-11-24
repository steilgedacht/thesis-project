"""Fit a trained Neural ODE to a NEW lesion by learning only its embedding.

Protocol (honest new-patient test):
    * The lesion must not be in the training set. Default: YG_69FMUN9PHHKE_2,
      which MRI_Dataloader.cache_lesion_trajectories_from_n_scans skips and
      whose patient has no other trained lesion.
    * Support = the first scans of the lesion, held out = the last
      ``--heldout-scans`` scans. Only the support scans are used for fitting.
    * The embedding is fitted in two phases (as before): a convex combination
      of the TRAINED embeddings (238 rows; the table has ~133k rows, the rest
      were never trained), then a direct fine-tune of that vector.
    * The region prior gets a box from the support scans only.
    * Evaluation: held-out (and support) scans on the same coarse-to-fine
      uniform grid as the training validation / validation_arena.py.

The protocol lives in utils/new_lesion_fit.py and is shared with the
embedding test of validation_arena.py.

Usage:
    python new_patient_fit.py --config configs/06_neural_ode/config_fixes_round2.py \
        --run-id 315faa8e61c74cceaced1bc78e9497a0
"""
import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import model_neural_ode_translation_validation, model_neural_ode_validation
from utils.dataloader import LesionDataset, Plotting_LesionDataset
from utils.grid_validation import evaluator_from_config
from utils.loss_bce_dice import Loss_BCE_Dice
from utils.mri_dataloader import MRI_Dataloader
from utils.new_lesion_fit import (
    HELDOUT_SCANS, NEW_LESION, PROTOCOL, basis_rows, global_lesion_id, lesion_dates,
    split_support_heldout, support_box, training_lesion_ids, uniform_coefficients,
)
from utils.train_plotting import plot_contanct_sheet, plot_lesion_time_evolution

DEFAULT_RUN_ID = "315faa8e61c74cceaced1bc78e9497a0"  # ODE + fixes, round 2 (6 GB)
DEFAULT_LESION = f"{NEW_LESION[0]}_{NEW_LESION[1]}"


def load_config(config_path: str):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()


def add_base_configurations(config):
    with open("configs/base_config.json", "r") as f:
        base_config = json.load(f)
    for key, value in base_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config


def load_mlflow_model(model_uri: str, device):
    validation_modules = {
        "utils.model_neural_ode": model_neural_ode_validation,
        "utils.model_neural_ode_translation": model_neural_ode_translation_validation,
    }
    original_modules = {name: sys.modules.get(name) for name in validation_modules}
    sys.modules.update(validation_modules)
    try:
        return mlflow.pytorch.load_model(model_uri, map_location=device)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _project_onto_simplex(values):
    """Project values onto the non-negative vectors whose entries sum to one."""
    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(1, values.numel() + 1, device=values.device, dtype=values.dtype)
    active = sorted_values - (cumulative - 1) / indices > 0
    last_active = torch.nonzero(active, as_tuple=False)[-1, 0]
    threshold = (cumulative[last_active] - 1) / (last_active + 1)
    return torch.clamp(values - threshold, min=0)


# ---------------------------------------------------------------------------
# Embedding fit
# ---------------------------------------------------------------------------
def fit_embedding(model, train_loader, basis_embeddings, lesion_idx, epochs, combination_epochs, lr, device):
    """Two-phase embedding fit; returns the fitted vector. Model weights stay frozen."""
    if epochs < 0 or combination_epochs < 0:
        raise ValueError("epochs and combination_epochs must be non-negative")
    combination_phase_epochs = min(combination_epochs, epochs)
    direct_phase_epochs = epochs - combination_phase_epochs
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Start at the mean of the trained embeddings (uniform combination).
    coefficients = torch.nn.Parameter(uniform_coefficients(basis_embeddings.shape[0], device))
    criterion = Loss_BCE_Dice()
    global_step = 0

    def train_phase(phase_name, phase_epochs, optimizer, embedding_for_batch, project_coefficients=False):
        nonlocal global_step
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25)
        hook = model.patient_embedding.register_forward_hook(
            lambda _module, _inputs, output: embedding_for_batch(output)
        )
        try:
            for phase_epoch in range(phase_epochs):
                total_loss = 0.0
                model.train()  # same mode as in training (region prior off if trained so)
                for coords, labels, _ in tqdm(
                    train_loader, desc=f"{phase_name} epoch {phase_epoch + 1}/{phase_epochs}", leave=False
                ):
                    if coords.shape[1] == 0:
                        continue
                    coords = coords.to(device)
                    labels = labels.to(device).unsqueeze(-1)
                    patient_idx = torch.full((coords.shape[0],), lesion_idx, dtype=torch.long, device=device)

                    optimizer.zero_grad()
                    loss = criterion(model(coords, patient_idx), labels)
                    loss.backward()
                    optimizer.step()
                    if project_coefficients:
                        with torch.no_grad():
                            coefficients.copy_(_project_onto_simplex(coefficients))
                    scheduler.step(metrics=loss.item())
                    total_loss += loss.item()
                    mlflow.log_metrics({
                        "training_loss": loss.item(),
                        "training_loss_bce": criterion.loss_bce.item(),
                        "training_loss_dice": criterion.loss_dice.item(),
                        f"{phase_name}_training_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }, step=global_step)
                    global_step += 1
                model.eval()
                if (phase_epoch + 1) % 10 == 0:
                    print(f"{phase_name} epoch {phase_epoch + 1}/{phase_epochs}, "
                          f"loss {total_loss / max(len(train_loader), 1):.4f}")
        finally:
            hook.remove()

    train_phase(
        "linear_combination", combination_phase_epochs,
        optim.AdamW([coefficients], lr=lr, weight_decay=1e-6),
        lambda output: (coefficients @ basis_embeddings).unsqueeze(0).expand_as(output),
        project_coefficients=True,
    )
    with torch.no_grad():
        fitted = coefficients @ basis_embeddings
    mlflow.log_metrics({
        "coefficient_max": coefficients.max().item(),
        "coefficient_effective_count": (1.0 / (coefficients ** 2).sum()).item(),
    })

    direct_embedding = torch.nn.Parameter(fitted.detach().clone())
    train_phase(
        "direct_embedding", direct_phase_epochs,
        optim.AdamW([direct_embedding], lr=lr * 0.01, weight_decay=1e-6),
        lambda output: direct_embedding.unsqueeze(0).expand_as(output),
    )
    return direct_embedding.detach().clone()


# ---------------------------------------------------------------------------
# Evaluation on the validation grid
# ---------------------------------------------------------------------------
def evaluate(model, evaluator, trj, dates, lesion_idx):
    rows = []
    for date in dates:
        result = evaluator.evaluate(model, trj, date, lesion_idx)
        rows.append(dict(
            date=date,
            dice=result.dice_score,
            volume_error_cm3=evaluator.wrongly_predicted_volume_cm3(result),
            target_cm3=result.target_count * evaluator.grid_point_volume_cm3,
            predicted_cm3=result.predicted_count * evaluator.grid_point_volume_cm3,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Fit a trained NODE to an unseen lesion (embedding only).")
    parser.add_argument("--config", default="configs/06_neural_ode/config_fixes_round2.py")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID, help="MLflow run of the trained NODE")
    parser.add_argument("--lesion", default=DEFAULT_LESION, help="<patient_id>_<label_id> of an unseen lesion")
    parser.add_argument("--heldout-scans", type=int, default=HELDOUT_SCANS, help="last scans used only for evaluation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--combination-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=20)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5124")
    parser.add_argument("--allow-training-lesion", action="store_true",
                        help="permit a lesion from the training set (not a new-patient test)")
    args = parser.parse_args()

    config = add_base_configurations(load_config(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("Lesion_Embedding")

    patient_id, label_id = args.lesion.rsplit("_", 1)
    mri_dataloader = MRI_Dataloader()
    trj = mri_dataloader.find_lesion_trajectory(patient_id, int(label_id))
    if trj is None:
        raise SystemExit(f"Lesion {args.lesion} not found.")
    support_dates, heldout_dates = split_support_heldout(lesion_dates(trj), args.heldout_scans)
    # Support-only copy: same first date (same time origin), later scans hidden.
    support_trj = copy.copy(trj)
    support_trj.allowed_dates = list(support_dates)

    # Training lesions -> rows of the embedding table that were actually trained.
    training_ids = training_lesion_ids(
        config.lesion_trajectories_with_more_than_n_scans, config.use_only_growing_lesions
    )
    lesion_idx = global_lesion_id(trj)
    if lesion_idx in training_ids and not args.allow_training_lesion:
        raise SystemExit(f"{args.lesion} is a training lesion; use an unseen lesion "
                         "(or --allow-training-lesion).")

    model = load_mlflow_model(f"runs:/{args.run_id}/lesion_ode_model", device).to(device)
    if lesion_idx >= model.patient_embedding.num_embeddings:
        raise SystemExit(f"Embedding id {lesion_idx} is outside the model's table.")
    rows = basis_rows(model, model.patient_embedding, training_ids, exclude_id=lesion_idx)
    basis = model.patient_embedding.weight.detach()[rows.to(device)].clone()

    prior = getattr(model, "region_prior", None)
    if prior is not None:
        box = support_box(trj, support_dates)
        if box is not None:
            prior.set_box(lesion_idx, box)

    train_dataset = LesionDataset(
        [support_trj] * args.samples_per_epoch,
        device="cpu",
        dialation_iterations=config.dialation_iterations,
        background_samples=config.background_samples,
        oversample_positive_samples=getattr(config, "oversample_positive_samples", True),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    with mlflow.start_run(run_name="embedding learning", tags={"type": "new_patient_fit"}):
        mlflow.log_params({
            "model_run_id": args.run_id, "lesion": args.lesion, "lesion_idx": lesion_idx,
            "support_dates": ",".join(support_dates), "heldout_dates": ",".join(heldout_dates),
            "basis_size": len(rows), "protocol": PROTOCOL, "epochs": args.epochs,
            "combination_epochs": args.combination_epochs, "lr": args.lr,
            "batch_size": args.batch_size, "samples_per_epoch": args.samples_per_epoch,
            "region_prior_box": prior is not None,
        })

        embedding = fit_embedding(model, train_loader, basis, lesion_idx, args.epochs,
                                  args.combination_epochs, args.lr, device)
        with torch.no_grad():
            model.patient_embedding.weight[lesion_idx].copy_(embedding)
        model.eval()

        evaluator = evaluator_from_config(config)
        with torch.no_grad():
            heldout = evaluate(model, evaluator, trj, heldout_dates, lesion_idx)
            support = evaluate(model, evaluator, trj, support_dates, lesion_idx)

        print(f"\n{args.lesion}: fitted on {len(support_dates)} scans, evaluated on {len(heldout_dates)} held-out")
        print(f"{'date':12s} {'split':8s} {'NODE dice':>9s} {'pred cm3':>9s} {'target cm3':>10s}")
        for split, rows in (("support", support), ("heldout", heldout)):
            for r in rows:
                print(f"{r['date']:12s} {split:8s} {r['dice']:9.3f} "
                      f"{r['predicted_cm3']:9.2f} {r['target_cm3']:10.2f}")
        metrics = {
            "new_patient_heldout_dice_score": float(np.mean([r["dice"] for r in heldout])),
            "new_patient_heldout_wrongly_predicted_volume_cm3": float(np.mean([r["volume_error_cm3"] for r in heldout])),
            "new_patient_support_dice_score": float(np.mean([r["dice"] for r in support])),
        }
        for i, r in enumerate(heldout, 1):
            metrics[f"new_patient_heldout_{i}_dice_score"] = r["dice"]
        mlflow.log_metrics(metrics)
        print({k: round(v, 3) for k, v in metrics.items()})

        # Plotting_LesionDataset keeps only its plotting list, so set the lesion
        # afterwards; __getitem__ overwrites embedding_id, so set it after that.
        plotting = Plotting_LesionDataset([trj], device="cpu")
        plotting.trajectories = [trj]
        plot_trj = plotting[0]
        plot_trj.embedding_id = np.int64(lesion_idx)
        with torch.no_grad():
            plot_contanct_sheet(model, args.epochs, plot_trj, config, final_side_length=True)
            plot_lesion_time_evolution(model, args.epochs, plot_trj, config, final_side_length=True)


if __name__ == "__main__":
    main()
