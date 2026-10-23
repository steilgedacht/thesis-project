import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import requests
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataloader_lstm import (
    LesionSequenceDataset,
    Plotting_LesionSequenceDataset,
    Validation_Extrapolation_LesionSequenceDataset,
    Validation_Interpolation_LesionSequenceDataset,
    sequence_collate_fn,
)
from utils.mri_dataloader import MRI_Dataloader
from utils.gradient_logging import gradient_statistics
from utils.train_plotting_lstm import (
    plot_contanct_sheet,
    plot_lesion_time_evolution_lstm,
)


_PHASE_TO_INT = {"autoencoder": 0, "mamba_only": 1, "joint": 2}


def check_if_mlflow_is_running(config):
    try:
        requests.get(f"{config.mlflow_tracking_uri}/version", timeout=2).raise_for_status()
    except requests.RequestException:
        uri_parts = config.mlflow_tracking_uri.split(":")
        if len(uri_parts) < 3:
            raise ValueError(
                f"Cannot start MLflow for invalid tracking URI: {config.mlflow_tracking_uri}"
            )
        host = uri_parts[1].split("//")[-1]
        port = uri_parts[2].split("/")[0]
        os.system(
            f"tmux new-session -d -s mlflow_server "
            f"'mlflow server --host {host} --port {port}'"
        )
        print("Starting MLflow in a new tmux session...")
        time.sleep(5)
        requests.get(f"{config.mlflow_tracking_uri}/version", timeout=5).raise_for_status()


def load_config(config_path):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load configuration from {config_file}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()


def add_base_configurations(config):
    with open("configs/base_config.json", "r", encoding="utf-8") as config_file:
        base_config = json.load(config_file)
    for key, value in base_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config


def get_training_phase(epoch, config):
    autoencoder_epochs = getattr(config, "autoencoder_pretrain_epochs", 0)
    mamba_epochs = getattr(config, "mamba_only_epochs", 0)
    if epoch <= autoencoder_epochs:
        return "autoencoder"
    if epoch <= autoencoder_epochs + mamba_epochs:
        return "mamba_only"
    return "joint"


def apply_training_phase(model, phase):
    if phase == "autoencoder":
        model.unfreeze_autoencoder()
        model.freeze_temporal()
    elif phase == "mamba_only":
        model.freeze_autoencoder()
        model.unfreeze_temporal()
    else:
        model.unfreeze_autoencoder()
        model.unfreeze_temporal()


def load_autoencoder_weights(model, config):
    checkpoint_path = getattr(config, "load_autoencoder_weights_path", None)
    if not checkpoint_path:
        return False
    model.load_autoencoder_weights(checkpoint_path, map_location=config.device)
    print(f"Loaded encoder and decoder weights from {checkpoint_path}")
    return True


def save_autoencoder_weights(model, config):
    checkpoint_path = getattr(config, "save_autoencoder_weights_path", None)
    if not checkpoint_path:
        return
    model.save_autoencoder_weights(checkpoint_path)
    mlflow.log_artifact(checkpoint_path)
    print(f"Saved encoder and decoder weights to {checkpoint_path}")


def _validation_sample_resampling_extent(dataset, sample_idx, split):
    if split == "Valid Interpolation":
        trajectory = dataset.trajectories[sample_idx]
        trajectory_idx = sample_idx
        full_dates = list(
            trajectory.allowed_dates
            if hasattr(trajectory, "allowed_dates")
            else trajectory.dates
        )
        sample_id = f"{trajectory.patient_id}_{trajectory.label_id}"
        target_date = full_dates[
            int(dataset.validation_samples[sample_id]["interpolation"])
        ]
        history_dates = full_dates[: full_dates.index(target_date)] or [target_date]
    else:
        trajectory_idx, _, target_date = dataset.trajectories_expanded[sample_idx]
        trajectory = dataset._base_trajectories[trajectory_idx]
        full_dates = list(
            trajectory.allowed_dates
            if hasattr(trajectory, "allowed_dates")
            else trajectory.dates
        )
        sample_id = f"{trajectory.patient_id}_{trajectory.label_id}"
        held_out_dates = dataset.validation_samples[sample_id]["extrapolation_dates"]
        history_dates = [date for date in full_dates if date not in held_out_dates]

    resampled = dataset._load_and_resample_trajectory(
        trajectory_idx, history_dates + [target_date]
    )
    return float(resampled["extent"])


def validate_polation(model, data_loader, split, global_step, criterion, config):
    bce_loss = 0.0
    dice_loss = 0.0
    volume_error_cm3 = 0.0
    grid_size = np.asarray(data_loader.dataset.grid_size, dtype=np.float64)
    loss_time_pairs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(data_loader, desc=split, total=len(data_loader))
        ):
            if len(batch) == 6:
                history_grids, history_times, target_grid, target_time, patient_idx, days = batch
            else:
                history_grids, history_times, target_grid, target_time, patient_idx = batch
                days = None

            history_grids = history_grids.to(config.device)
            history_times = history_times.to(config.device)
            target_grid = target_grid.to(config.device)
            target_time = target_time.to(config.device).view(-1, 1)
            patient_idx = patient_idx.to(config.device)
            all_times = torch.cat([history_times, target_time], dim=1)

            predictions = model(
                history_grids, all_times, patient_idx=patient_idx, n_future=1
            )
            target_prediction = predictions[:, -1]
            batch_bce = criterion.loss_fn_1(target_prediction, target_grid).mean().item()
            batch_dice = criterion.dice_loss(target_prediction, target_grid).mean().item()
            bce_loss += batch_bce
            dice_loss += batch_dice

            batch_start = batch_idx * (getattr(data_loader, "batch_size", 1) or 1)
            for local_idx in range(target_prediction.shape[0]):
                extent_mm = _validation_sample_resampling_extent(
                    data_loader.dataset, batch_start + local_idx, split
                )
                voxel_volume_cm3 = np.prod(extent_mm / (grid_size - 1)) / 1000.0
                predicted_voxels = (
                    torch.sigmoid(target_prediction[local_idx]) > 0.5
                ).sum().item()
                target_voxels = (target_grid[local_idx] > 0.5).sum().item()
                volume_error_cm3 += (
                    abs(predicted_voxels - target_voxels) * voxel_volume_cm3
                )
            if days is not None and split == "Valid Extrapolation":
                loss_time_pairs.append((batch_dice, days))

    num_batches = len(data_loader)
    avg_bce = bce_loss / num_batches if num_batches else 0.0
    avg_dice = dice_loss / num_batches if num_batches else 0.0
    metric_prefix = split.lower().replace(" ", "_")
    mlflow.log_metrics(
        {
            f"{metric_prefix}_bce_loss": avg_bce,
            f"{metric_prefix}_dice_loss": avg_dice,
            f"{metric_prefix}_total_loss": avg_bce + avg_dice,
            f"{metric_prefix}_dice_score": 1 - avg_dice,
            f"{metric_prefix}_wrongly_predicted_volume_cm3": volume_error_cm3,
        },
        step=global_step,
    )
    print(f"{split} wrongly predicted volume: {volume_error_cm3:.6f} cm^3")

    if split == "Valid Extrapolation" and loss_time_pairs:
        import matplotlib.pyplot as plt

        values = np.asarray(
            [[loss, torch.as_tensor(days).float().mean().item()] for loss, days in loss_time_pairs]
        )
        x = 180 - values[:, 1]
        y = 1 - values[:, 0]
        if len(x) > 1 and np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, float(y.mean())
        plt.figure()
        plt.scatter(x, y, label="Extrapolation Scan")
        plt.plot(
            np.linspace(1, 180, 200),
            slope * np.linspace(1, 180, 200) + intercept,
            color="red",
            linestyle="--",
            label=f"Linear Fit (Slope = {slope:.6f})",
        )
        plt.title("Extrapolation Dice Score after t_days in the last 180 days")
        plt.ylabel("Dice Score")
        plt.ylim(0, 1)
        plt.xlabel("t [days]")
        plt.xlim(1, 183)
        plt.xticks(list(range(0, 181, 20)))
        plt.legend()
        output_path = f"/tmp/Extrapolation_Dice_Score_Distribution_Step{global_step}.png"
        plt.savefig(output_path)
        mlflow.log_artifact(output_path)
        plt.close()


def train_mamba(
    model,
    train_loader,
    valid_interpolation_loader,
    valid_extrapolation_loader,
    plotting_dataset,
    config,
):
    model.to(config.device)
    autoencoder_loaded = load_autoencoder_weights(model, config)
    freeze_loaded_autoencoder = (
        autoencoder_loaded and getattr(config, "freeze_loaded_autoencoder", True)
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.scheduler_eta_min
    )
    criterion = config.loss_fn
    accumulation_steps = max(1, config.gradient_accumulation_steps)
    losses = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, config.epochs + 1):
        phase = get_training_phase(epoch, config)
        if phase != get_training_phase(epoch - 1, config):
            save_autoencoder_weights(model, config)
        if freeze_loaded_autoencoder:
            phase = "mamba_only"
        apply_training_phase(model, phase)
        model.train()
        total_loss = 0.0

        for batch_idx, (grids, times, patient_idx, lengths) in enumerate(
            tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch}/{config.epochs} [{phase}]",
            )
        ):
            grids = grids.to(config.device)
            times = times.to(config.device)
            patient_idx = patient_idx.to(config.device)
            lengths = lengths.to(config.device)

            if phase == "autoencoder":
                predictions = model.forward_autoencoder(grids)
                targets = grids
                valid_steps = (
                    torch.arange(predictions.shape[1], device=config.device)[None, :]
                    < lengths[:, None]
                )
            else:
                predictions = model(grids, times, patient_idx)
                targets = grids[:, 1:]
                valid_steps = (
                    torch.arange(predictions.shape[1], device=config.device)[None, :]
                    < (lengths - 1).clamp_min(0)[:, None]
                )

            predictions = predictions[valid_steps]
            targets = targets[valid_steps]
            loss = criterion(predictions, targets)
            unscaled_loss = loss.detach().item()
            (loss / accumulation_steps).backward()
            grad_metrics = (
                gradient_statistics(model)
                if batch_idx % config.train_log_interval == 0
                else {}
            )
            total_loss += unscaled_loss

            should_step = (
                (batch_idx + 1) % accumulation_steps == 0
                or (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=config.max_grad_norm_clip,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if batch_idx % config.train_log_interval == 0:
                metrics = {
                    **grad_metrics,
                    "training_loss": unscaled_loss,
                    "wrongly_predicted_total_volume": (
                        (predictions > 0).sum().item()
                        - (targets > 0.5).sum().item()
                    ),
                    "training_phase": _PHASE_TO_INT[phase],
                }
                metrics.update(criterion.loss_to_report)
                mlflow.log_metrics(metrics, step=global_step)

            global_step += 1
            if (
                config.device.startswith("cuda")
                and batch_idx % config.train_delete_cache_interval == 0
            ):
                torch.cuda.empty_cache()

        if config.only_train:
            scheduler.step()
            losses.append(total_loss / max(1, len(train_loader)))
            continue

        model.eval()
        if epoch % config.validation_interval == 0 and phase != "autoencoder":
            validate_polation(
                model,
                valid_extrapolation_loader,
                "Valid Extrapolation",
                global_step,
                criterion,
                config,
            )
            validate_polation(
                model,
                valid_interpolation_loader,
                "Valid Interpolation",
                global_step,
                criterion,
                config,
            )

        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        scheduler.step()
        average_loss = total_loss / max(1, len(train_loader))
        losses.append(average_loss)
        if epoch % config.print_loss_interval == 0:
            print(
                f"Epoch {epoch}/{config.epochs}, Loss: {average_loss:.6f}, "
                f"Phase: {phase}"
            )

    mlflow.pytorch.log_model(
        model,
        name=config.model_save_name,
        serialization_format="pickle",
    )
    save_autoencoder_weights(model, config)

    if not config.only_train:
        model.eval()
        with torch.no_grad():
            for trajectory in plotting_dataset:
                plot_contanct_sheet(
                    model, epoch, trajectory, config, final_side_length=True
                )
                plot_lesion_time_evolution_lstm(
                    model, epoch, trajectory, config, final_side_length=True
                )
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a latent-space Mamba lesion model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/07_mamba/config.py",
        help="Path to the Mamba configuration file.",
    )
    args = parser.parse_args()
    config = add_base_configurations(load_config(args.config))
    print(f"Using configuration from: {args.config}")

    check_if_mlflow_is_running(config)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(
            n_scans=config.lesion_trajectories_with_more_than_n_scans,
            only_growing=config.use_only_growing_lesions,
        )
        if mri_dataloader.cache_lesion_trajectories is None:
            raise RuntimeError("No lesion trajectories were cached")

        trajectories = (
            mri_dataloader.cache_lesion_trajectories
            * config.training_dataset_samples_duplication_factor
        )
        train_dataset = LesionSequenceDataset(
            trajectories,
            cache_dir="/tmp/lesion_sequence_cache",
            **config.dataset_params,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batchsize,
            shuffle=True,
            collate_fn=sequence_collate_fn,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor if config.num_workers else None,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
        )
        validation_trajectories = mri_dataloader.cache_lesion_trajectories
        validation_kwargs = dict(config.dataset_params)
        validation_kwargs["cache_dir"] = None
        valid_interpolation_dataset = Validation_Interpolation_LesionSequenceDataset(
            validation_trajectories, **validation_kwargs
        )
        valid_interpolation_loader = DataLoader(
            valid_interpolation_dataset, shuffle=False
        )
        valid_extrapolation_dataset = Validation_Extrapolation_LesionSequenceDataset(
            validation_trajectories, **validation_kwargs
        )
        valid_extrapolation_loader = DataLoader(
            valid_extrapolation_dataset, shuffle=False
        )
        plotting_dataset = Plotting_LesionSequenceDataset(
            validation_trajectories.copy(), **validation_kwargs
        )

        model = config.model(trajectories=trajectories, **config.model_params)
        mlflow.log_params(
            {
                "dataset_size": len(train_dataset),
                "loss_function": type(config.loss_fn).__name__,
                "model_type": type(model).__name__,
                **{
                    key: str(getattr(config, key))
                    for key in dir(config)
                    if not key.startswith("_")
                    and isinstance(getattr(config, key), (str, int, float, bool))
                },
            }
        )
        mlflow.log_artifact(args.config)
        mlflow.log_artifact(os.path.basename(__file__))
        losses = train_mamba(
            model,
            train_loader,
            valid_interpolation_loader,
            valid_extrapolation_loader,
            plotting_dataset,
            config,
        )
        if losses:
            mlflow.log_metric("final_train_loss", losses[-1])
