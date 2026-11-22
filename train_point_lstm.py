"""Train the Point-LSTM (observation-conditioned lesion forecaster).

    python train_point_lstm.py --config configs/09_point_lstm/config.py
    python train_point_lstm.py --config configs/09_point_lstm/config_sdf.py
"""

import argparse
import importlib.util
import json
import os
import random
import socket
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import requests
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloader_point_lstm import (
    PointLSTMDataset,
    PointLSTMValidationDataset,
    ScanCache,
    batch_to_device,
    collate_history,
    evaluation_history,
    history_arrays,
    lesion_key,
    point_lstm_collate,
)
from utils.grid_validation import evaluator_from_config, grid_validation_metrics
from utils.mri_dataloader import MRI_Dataloader
from utils.paths import DatasetPaths
from utils.train_plotting_point_lstm import plot_point_lstm_time_evolution

print(f"Hostname: {socket.gethostname()}")


def check_if_mlflow_is_running(config):
    try:
        requests.get(f"{config.mlflow_tracking_uri}/version", timeout=5)
    except Exception:
        import time
        os.system("tmux new-session -d -s mlflow_server 'mlflow server --port 5124'")
        print("Starting MLflow in a new tmux session...")
        time.sleep(5)


def load_config(config_path):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Config()


def add_base_configurations(config):
    with open("configs/base_config.json", "r") as file:
        for key, value in json.load(file).items():
            if not hasattr(config, key):
                setattr(config, key, value)
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hard_dice(logits, labels):
    predicted = (logits > 0).float()
    intersection = (predicted * labels).sum()
    return float((2 * intersection + 1e-6) / (predicted.sum() + labels.sum() + 1e-6))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, name, criterion, config, global_step):
    model.eval()
    soft, hard, days_and_dice = [], [], []
    for batch in tqdm(loader, desc=name):
        batch = batch_to_device(batch, config.device)
        logits = model(batch["coords"], batch["patient_idx"], batch["history"], sdf=batch["sdf"])
        for b in range(logits.shape[0]):
            soft.append(1.0 - criterion.dice_loss(logits[b], batch["labels"][b]).item())
            hard.append(hard_dice(logits[b], batch["labels"][b]))
            if "days_from_endpoint" in batch and batch["days_from_endpoint"][b] >= 0:
                days_and_dice.append((float(batch["days_from_endpoint"][b]), hard[-1]))
    prefix = name.lower().replace(" ", "_")
    mlflow.log_metrics(
        {f"{prefix}_dice_score": float(np.mean(soft)),
         f"{prefix}_dice_loss": 1.0 - float(np.mean(soft)),
         f"{prefix}_hard_dice_score": float(np.mean(hard))},
        step=global_step,
    )
    model.train()
    return days_and_dice


_GRID_EVALUATOR = None


@torch.no_grad()
def validate_grid(model, loader, name, config, global_step):
    """Coarse-to-fine uniform-grid Dice, matching validation_arena.py (see utils/grid_validation.py).

    History and latent are computed once per case exactly like the arena's
    _infer_point_lstm; only the decoder is queried on the grid points.
    """
    global _GRID_EVALUATOR
    if _GRID_EVALUATOR is None:  # one per run: fine ground truth is cached across epochs
        _GRID_EVALUATOR = evaluator_from_config(config)
    evaluator = _GRID_EVALUATOR
    dataset = loader.dataset
    model.eval()

    lesion_ids, results, days_and_dice = [], [], []
    for trj, target_date, days in tqdm(dataset.cases, desc=name):
        history_dates = evaluation_history(
            trj, target_date, dataset.validation_samples, model.max_history_len
        )
        entries = [dataset.cache.get(trj, d) for d in history_dates]
        rng = np.random.default_rng(
            dataset.seed + zlib.crc32(f"{lesion_key(trj)}|{target_date}".encode())
        )
        history = batch_to_device(
            collate_history([history_arrays(entries, model.history_points, rng, dataset.ring_mm)]),
            config.device,
        )
        patient_idx = torch.as_tensor([int(dataset.get_embedding_id(trj))], device=config.device)
        target_time = torch.tensor(
            [evaluator.target_time(trj, target_date)], dtype=torch.float32, device=config.device
        )
        z = model.latent(model.encode_history(history), target_time, patient_idx)

        def query(xyz, time_point, z=z, last=entries[-1]):
            sdf = None
            if model.use_sdf_prior:
                sdf = torch.from_numpy(last.sdf_at_model(xyz)).view(1, -1, 1).to(config.device)
            return model.decode(evaluator.coords_tensor(xyz, time_point), z, sdf)

        result = evaluator.evaluate_query(query, trj, target_date)
        lesion_ids.append(lesion_key(trj))
        results.append(result)
        if days >= 0:
            days_and_dice.append((float(days), result.dice_score))

    metrics, _ = grid_validation_metrics(
        name.lower().replace(" ", "_"), lesion_ids, results, evaluator,
        extrapolation=(dataset.split == "extrapolation"),
    )
    mlflow.log_metrics(metrics, step=global_step)
    model.train()
    return days_and_dice


def run_validation(model, loader, name, criterion, config, global_step):
    """Dispatch on config.validation_mode: "grid" (default) or "sampled" (legacy)."""
    mode = getattr(config, "validation_mode", "grid")
    if mode == "grid":
        return validate_grid(model, loader, name, config, global_step)
    if mode == "sampled":
        return validate(model, loader, name, criterion, config, global_step)
    raise ValueError(f"Unknown validation_mode: {mode!r} (expected 'grid' or 'sampled').")


# ---------------------------------------------------------------------------
# Qualitative plot: axial slice through the target lesion
# ---------------------------------------------------------------------------
@torch.no_grad()
def plot_slices(model, dataset, config, epoch, n_cases):
    model.eval()
    cases = dataset.cases[:n_cases]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 3, figsize=(9, 3 * len(cases)), squeeze=False)
    for row, (trj, target_date, days) in enumerate(cases):
        history_dates = evaluation_history(trj, target_date, dataset.validation_samples, model.max_history_len)
        entries = [dataset.cache.get(trj, d) for d in history_dates]
        target, last = dataset.cache.get(trj, target_date), entries[-1]
        reference = target if len(target.positive_voxels) else last
        boxes = [e.crop_box for e in (target, last) if e.sdf_q is not None]
        if boxes:
            lo = np.min([b[0] for b in boxes], axis=0)
            hi = np.max([b[1] for b in boxes], axis=0)
        else:
            lo, hi = np.zeros(3, dtype=np.int64), np.asarray(target.shape)
        z_slice = int(np.median(reference.positive_voxels[:, 2])) if len(reference.positive_voxels) else (lo[2] + hi[2]) // 2
        ii, jj = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
        voxels = np.stack([ii.ravel(), jj.ravel(), np.full(ii.size, z_slice)], axis=1)

        xyz = target.voxels_to_model(voxels)
        coords = np.concatenate([xyz, np.full((len(xyz), 1), target.time, np.float32)], axis=1)
        rng = np.random.default_rng(0)
        history = collate_history([history_arrays(entries, model.history_points, rng, dataset.ring_mm)])
        history = batch_to_device(history, config.device)
        sdf = torch.from_numpy(last.sdf_at_model(xyz))[None, :, None].to(config.device)
        patient_idx = torch.as_tensor([int(dataset.get_embedding_id(trj))], device=config.device)
        logits = model(torch.from_numpy(coords)[None].to(config.device), patient_idx, history, sdf=sdf)
        probability = torch.sigmoid(logits).reshape(ii.shape).cpu().numpy()

        panels = [
            (last.label_at_voxels(voxels).reshape(ii.shape), f"last observed ({history_dates[-1]})"),
            (probability, f"prediction (+{target.time - last.time:.0f} d)"),
            (target.label_at_voxels(voxels).reshape(ii.shape), f"target ({target_date})"),
        ]
        for col, (image, title) in enumerate(panels):
            axes[row, col].imshow(image.T, cmap="magma", vmin=0, vmax=1, origin="lower")
            axes[row, col].contour(panels[2][0].T, levels=[0.5], colors="cyan", linewidths=0.7)
            axes[row, col].set_title(f"{lesion_key(trj)}\n{title}" if col == 0 else title, fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    output_path = f"/tmp/point_lstm_slices_{dataset.split}_epoch{epoch:03d}.png"
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    mlflow.log_artifact(output_path)
    model.train()


def plot_extrapolation_by_horizon(days_and_dice, global_step, ylabel="hard Dice (sampled points)"):
    if len(days_and_dice) < 2:
        return
    days, dice = np.asarray(days_and_dice).T
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(days, dice, s=12)
    ax.set_xlabel("days before trajectory end (T minus)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    output_path = f"/tmp/point_lstm_extrapolation_horizon_step{global_step}.png"
    fig.savefig(output_path)
    plt.close(fig)
    mlflow.log_artifact(output_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def plot_time_evolutions(model, trajectories, dataset, config, epoch, final_side_length=False):
    """Time-evolution GIFs (as in utils/train_plotting.py) for the plotting lesions."""
    for trj in trajectories:
        plot_point_lstm_time_evolution(
            model, epoch, trj, dataset.cache, dataset.validation_samples,
            dataset.get_embedding_id(trj), config,
            final_side_length=final_side_length, ring_mm=dataset.ring_mm,
        )


def train(model, train_loader, valid_interpolation_loader, valid_extrapolation_loader, config,
          plotting_trajectories=()):
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.scheduler_eta_min
    )
    criterion = config.loss_fn
    model.to(config.device)
    global_step, losses = 0, []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader),
                             desc=f"Epoch {epoch}/{config.epochs}"):
            batch = batch_to_device(batch, config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["coords"], batch["patient_idx"], batch["history"], sdf=batch["sdf"])
            loss = criterion(logits, batch["labels"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm_clip)
            optimizer.step()
            total_loss += loss.item()

            if i % config.train_log_interval == 0:
                metrics = {
                    "training_loss": loss.item(),
                    **criterion.loss_to_report,
                    "grad_norm": float(grad_norm),
                    "training_hard_dice": hard_dice(logits.detach(), batch["labels"]),
                }
                if model.use_sdf_prior:
                    metrics["sdf_prior_gain"] = float(model.sdf_prior_gain.detach())
                mlflow.log_metrics(metrics, step=global_step)
            global_step += 1

        losses.append(total_loss / max(1, len(train_loader)))
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        mlflow.log_metric("epoch_training_loss", losses[-1], step=global_step)
        scheduler.step()
        print(f"Epoch {epoch}: loss {losses[-1]:.5f}")

        if config.only_train:
            continue
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            days_and_dice = run_validation(model, valid_extrapolation_loader, "Valid Extrapolation",
                                           criterion, config, global_step)
            run_validation(model, valid_interpolation_loader, "Valid Interpolation",
                           criterion, config, global_step)
            if epoch % config.plot_validation_interval == 0 or epoch == config.epochs:
                plot_extrapolation_by_horizon(
                    days_and_dice, global_step,
                    "hard Dice (uniform grid)" if getattr(config, "validation_mode", "grid") == "grid"
                    else "hard Dice (sampled points)",
                )
                for loader in (valid_extrapolation_loader, valid_interpolation_loader):
                    plot_slices(model, loader.dataset, config, epoch, config.n_monitoring_samples_to_visualize)
                if epoch != config.epochs:  # the final epoch is plotted at full size below
                    plot_time_evolutions(model, plotting_trajectories,
                                         valid_extrapolation_loader.dataset, config, epoch)

    mlflow.pytorch.log_model(model, name=config.model_save_name, serialization_format="pickle")
    plot_time_evolutions(model, plotting_trajectories, valid_extrapolation_loader.dataset,
                         config, config.epochs, final_side_length=True)
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train the Point-LSTM.")
    parser.add_argument("--config", default="configs/09_point_lstm/config.py")
    args = parser.parse_args()
    config = add_base_configurations(load_config(args.config))
    print(f"Using configuration from: {args.config}")
    seed_everything(config.seed)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    check_if_mlflow_is_running(config)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(
            n_scans=config.lesion_trajectories_with_more_than_n_scans,
            only_growing=config.use_only_growing_lesions,
        )
        base_trajectories = mri_dataloader.cache_lesion_trajectories
        assert base_trajectories, "No lesions cached"

        # One shared scan cache, filled before the DataLoader forks so that
        # all workers share it copy-on-write.
        dataset_params = dict(config.dataset_params)
        cache = ScanCache(sdf_clip_mm=config.model_params["sdf_clip_mm"])
        cache.preload(base_trajectories, processes=config.preload_processes)
        dataset_params["max_history_len"] = config.model_params["max_history_len"]
        dataset_params["history_points"] = config.model_params["history_points"]

        train_dataset = PointLSTMDataset(
            base_trajectories * config.training_dataset_samples_duplication_factor,
            cache=cache, **dataset_params,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batchsize, shuffle=True,
            collate_fn=point_lstm_collate, num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor if config.num_workers else None,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            drop_last=True,
        )
        validation_loaders = {
            split: DataLoader(
                PointLSTMValidationDataset(base_trajectories, split=split, cache=cache, **dataset_params),
                batch_size=config.validation_batchsize, shuffle=False,
                collate_fn=point_lstm_collate, num_workers=0,
            )
            for split in ("interpolation", "extrapolation")
        }

        num_patients = int(max(train_dataset.get_embedding_id(t) for t in base_trajectories)) + 1
        model_params = dict(config.model_params)
        model_params.setdefault("num_patients", num_patients)
        model = config.model(**model_params)

        mlflow.log_params({
            "dataset_size": len(train_dataset),
            "n_lesions": len(base_trajectories),
            "n_cached_scans": len(cache),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "valid_interpolation_cases": len(validation_loaders["interpolation"].dataset),
            "valid_extrapolation_cases": len(validation_loaders["extrapolation"].dataset),
            "loss_function": type(config.loss_fn).__name__,
            **{k: getattr(config, k) for k in dir(config) if not k.startswith("_")},
        })
        mlflow.log_artifact(args.config)
        for extra in getattr(config, "extra_config_files", []):
            mlflow.log_artifact(extra)
        for source in ("train_point_lstm.py", "utils/model_point_lstm.py", "utils/dataloader_point_lstm.py"):
            mlflow.log_artifact(source)

        with open(DatasetPaths().plotting_samples(), encoding="utf-8") as file:
            plotting_keys = {f"{s['Patient']}_{s['Lesion']}" for s in json.load(file)}
        plotting_trajectories = list(
            {lesion_key(t): t for t in base_trajectories if lesion_key(t) in plotting_keys}.values()
        )

        losses = train(model, train_loader, validation_loaders["interpolation"],
                       validation_loaders["extrapolation"], config,
                       plotting_trajectories=plotting_trajectories)
        if losses:
            mlflow.log_metric("final_train_loss", losses[-1])


if __name__ == "__main__":
    main()
