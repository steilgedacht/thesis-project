import importlib.util
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader_lstm import (
    LesionSequenceDataset,
    Validation_Extrapolation_LesionSequenceDataset,
    Validation_Interpolation_LesionSequenceDataset,
    Plotting_LesionSequenceDataset,
    sequence_collate_fn,
)
from utils.train_plotting_lstm import *
import argparse
import requests
import json
import os

def check_if_mlflow_is_running(config):
    try:
        requests.get(f"{config.mlflow_tracking_uri}/version")
    except Exception as e:
        import shutil
        import time
        shutil.os.system(f"tmux new-session -d -s mlflow_server 'mlflow server --host {config.mlflow_tracking_uri.split(':')[1].split('/')[2]} --port {config.mlflow_tracking_uri.split(':')[2]}'")
        print("Starting Mlflow in a new tmux session...")
        time.sleep(5)

def load_config(config_path: str):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()


# ---------------------------------------------------------------------------
# Staged-training helpers
# ---------------------------------------------------------------------------
_PHASE_TO_INT = {"autoencoder": 0, "lstm_only": 1, "joint": 2}


def get_training_phase(epoch, config):
    """
    Decide which part of the model should be trained this epoch.

      'autoencoder' -> encoder+decoder only, reconstructing each individual
                        frame (no RNN involved). Lets the autoencoder learn to
                        actually represent small/localized structures before
                        an untrained, noisy RNN starts pushing gradients
                        through it.
      'lstm_only'   -> encoder+decoder frozen, only the temporal model (LSTM
                        + fc_out [+ patient embedding]) is trained on top of
                        the now-stable latent space.
      'joint'       -> everything trainable together (original behaviour).

    Controlled by config.autoencoder_pretrain_epochs and
    config.lstm_only_epochs (both default to 0, i.e. pure joint training,
    if not set -- so this is backward compatible with old configs).
    """
    ae_epochs = getattr(config, "autoencoder_pretrain_epochs", 0)
    lstm_epochs = getattr(config, "lstm_only_epochs", 0)
    if epoch <= ae_epochs:
        return "autoencoder"
    elif epoch <= ae_epochs + lstm_epochs:
        return "lstm_only"
    return "joint"


def apply_training_phase(model, phase):
    """
    Set requires_grad flags on the model according to the phase. This is a
    no-op (always full 'joint' behaviour) for models that don't implement
    the staged-pretraining helpers (e.g. the plain grid-based LesionLSTM),
    so it's safe to call regardless of which model class is configured.
    """
    if not hasattr(model, "freeze_autoencoder"):
        return
    if phase == "autoencoder":
        model.unfreeze_autoencoder()
        model.freeze_temporal()
    elif phase == "lstm_only":
        model.freeze_autoencoder()
        model.unfreeze_temporal()
    else:
        model.unfreeze_autoencoder()
        model.unfreeze_temporal()


def load_autoencoder_weights(model, config):
    """Load configured encoder/decoder weights and report whether used."""
    checkpoint_path = getattr(config, "load_autoencoder_weights_path", None)
    if not checkpoint_path:
        return False
    if not hasattr(model, "load_autoencoder_weights"):
        raise ValueError("Configured autoencoder weights require an autoencoder LSTM model")
    model.load_autoencoder_weights(checkpoint_path, map_location=config.device)
    print(f"Loaded encoder and decoder weights from {checkpoint_path}")
    return True


def save_autoencoder_weights(model, config):
    """Save configured encoder/decoder weights, if requested."""
    checkpoint_path = getattr(config, "save_autoencoder_weights_path", None)
    if not checkpoint_path:
        return
    if not hasattr(model, "save_autoencoder_weights"):
        raise ValueError("Configured autoencoder weights require an autoencoder LSTM model")
    model.save_autoencoder_weights(checkpoint_path)
    mlflow.log_artifact(checkpoint_path)
    print(f"Saved encoder and decoder weights to {checkpoint_path}")


def validate_polation(data_loader, text, global_step, criterion):
    bce_loss = 0
    dice_loss = 0
    with torch.no_grad():
        for history_grids, history_times, target_grid, target_time, patient_idx in tqdm(
            data_loader, desc=text, total=len(data_loader)
        ):
            history_grids = history_grids.to(config.device)            # [1, T, 1, D, H, W]
            history_times = history_times.to(config.device)            # [1, T]
            target_grid = target_grid.to(config.device)                # [1, 1, D, H, W]
            target_time = target_time.to(config.device).view(1, 1)     # [1, 1]
            patient_idx = patient_idx.to(config.device)

            full_times = torch.cat([history_times, target_time], dim=1)  # [1, T+1]

            preds = model(history_grids, full_times, patient_idx=patient_idx, n_future=1)
            target_pred = preds[:, -1]  # [1, 1, D, H, W] logits for the held-out step

            # Use the same loss as training (BCE + Dice) for consistent monitoring
            bce_loss += criterion.loss_fn_1(target_pred, target_grid).mean().item()
            dice_loss += criterion.dice_loss(target_pred, target_grid).mean().item()

    avg_bce = bce_loss / len(data_loader)
    avg_dice = dice_loss / len(data_loader)
    
    mlflow.log_metrics({
        text.lower().replace(" ", "_") + "_bce_loss"  : avg_bce,
        text.lower().replace(" ", "_") + "_dice_loss" : avg_dice,
        text.lower().replace(" ", "_") + "_total_loss": avg_bce + avg_dice,
        text.lower().replace(" ", "_") + "_dice_score": 1 - avg_dice,
    }, step=global_step)

def train_lstm(
        model, 
        train_loader, 
        valid_interpolation_loader, 
        valid_extrapolation_loader,
        plotting_LesionDataset, 
        config,
    ):
    model.to(config.device)
    autoencoder_loaded = load_autoencoder_weights(model, config)
    freeze_loaded_autoencoder = (
        autoencoder_loaded and getattr(config, "freeze_loaded_autoencoder", True)
    )

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs, 
        eta_min=config.scheduler_eta_min
    )
    criterion = config.loss_fn

    losses = []
    
    global_step = epoch = 0

    for epoch in range(1, config.epochs + 1):
        total_loss = 0

        # ==== Decide / apply this epoch's training phase ====
        phase = get_training_phase(epoch, config)
        if phase != get_training_phase(epoch - 1, config):
            save_autoencoder_weights(model, config)
        if freeze_loaded_autoencoder:
            phase = "lstm_only"
        apply_training_phase(model, phase)
        if freeze_loaded_autoencoder:
            model.freeze_autoencoder()
        supports_staged_training = hasattr(model, "forward_autoencoder")

        # ==== Training Loop ====
        model.train()
        for i, (grids, times, patient_idx, lengths) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs} [{phase}]"):

            grids = grids.to(config.device)          # [1, T, 1, D, H, W]
            times = times.to(config.device)          # [1, T]
            patient_idx = patient_idx.to(config.device)
            lengths = lengths.to(config.device)

            if phase == "autoencoder" and supports_staged_training:
                # No RNN involved: reconstruct every observed frame independently.
                predictions = model.forward_autoencoder(grids)   # [1, T, 1, D, H, W]
                targets = grids
                valid_steps = torch.arange(
                    predictions.shape[1], device=config.device
                )[None, :] < lengths[:, None]
            else:
                predictions = model(grids, times, patient_idx)   # [1, T-1, 1, D, H, W]
                targets = grids[:, 1:]                            # ground-truth next-frame at each step
                valid_steps = torch.arange(
                    predictions.shape[1], device=config.device
                )[None, :] < (lengths - 1).clamp_min(0)[:, None]

            predictions = predictions[valid_steps]
            targets = targets[valid_steps]

            loss = criterion(predictions, targets)

            if loss.requires_grad:  
                # 1. Scale the loss to average the gradients over the accumulation steps
                loss = loss / config.gradient_accumulation_steps
                
                # 2. Accumulate gradients (adds to existing .grad instead of overwriting)
                loss.backward()

            total_loss += loss.item() * config.gradient_accumulation_steps # Scale back up for logging

            # 3. Only step the optimizer every 'b' steps
            if (i + 1) % config.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_norm=config.max_grad_norm_clip,
                )
                optimizer.step()
                optimizer.zero_grad()

            if i % config.train_log_interval == 0:
                # predictions/targets are logits here; compare against 0 (== sigmoid(x) > 0.5),
                # not 0.5, since these are raw logits not probabilities.
                mlflow.log_metrics(
                    {
                        # Report unscaled loss for accurate tracking
                        "training_loss": loss.item() * config.gradient_accumulation_steps,
                        **criterion.loss_to_report,
                        "wrongly_predicted_total_volume": (predictions > 0).sum().item() - (targets > 0.5).sum().item(),
                        "training_phase": _PHASE_TO_INT[phase],
                    },
                    step=global_step
                )

            global_step += 1

            if i % config.train_delete_cache_interval == 0:
                del grids, times, patient_idx, targets, predictions, loss
                torch.cuda.empty_cache()
        if config.only_train: continue

        # ==== Evaluation Loop ====
        model.eval()

        # Extrapolation/interpolation validation exercises the RNN, which is
        # meaningless (frozen/untrained) during the pure autoencoder phase, so skip it then.
        if epoch % config.validation_interval == 0 and phase != "autoencoder":
            with torch.no_grad():
                # visualize_samples(model, epoch, monitoring_samples, train_loader, "train", config)
                # visualize_samples(model, epoch, monitoring_samples, valid_interpolation_loader, "valid", config)
                validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion)
                validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion)
                for trj in plotting_LesionDataset:
                    plot_lesion_time_evolution_lstm(model, epoch, trj, config)

        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % config.print_loss_interval == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.6f}, Phase: {phase}")

    # ==== Final Logging and Visualization ====
    
    mlflow.pytorch.log_model(
        model, 
        name=config.model_save_name, 
        serialization_format="pickle"
    )

    save_autoencoder_weights(model, config)

    with torch.no_grad():
        for trj in plotting_LesionDataset:  
            plot_lesion_time_evolution_lstm(model, epoch, trj, config, final_side_length=True)

    return losses

def add_base_configurations(config):
    with open("configs/base_config.json", "r") as f:
        base_config = json.load(f)
    for key, value in base_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description="Train a Lesion Trajectory model.")
    argument_parser.add_argument("--config", type=str, default="configs/00_default/config.py", help="Path to the configuration file.")
    args = argument_parser.parse_args()
    config = load_config(args.config)
    config = add_base_configurations(config)
    print(f"Using configuration from: {args.config}")

    check_if_mlflow_is_running(config)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(
            n_scans=config.lesion_trajectories_with_more_than_n_scans, 
            only_growing=config.use_only_growing_lesions
        )

        assert mri_dataloader.cache_lesion_trajectories is not None, "No Lesions cached"
            
        trajectories = mri_dataloader.cache_lesion_trajectories * config.training_dataset_samples_duplication_factor
        train_dataset = LesionSequenceDataset(
            trajectories, 
            cache_dir="/tmp/lesion_sequence_cache",
            **config.dataset_params
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batchsize, 
            shuffle=True, 
            collate_fn=sequence_collate_fn,
            num_workers=config.num_workers, 
            prefetch_factor=config.prefetch_factor, 
            pin_memory=config.pin_memory, 
            persistent_workers=config.persistent_workers
        )
        
        valid_interpolation_dataset = Validation_Interpolation_LesionSequenceDataset(
            mri_dataloader.cache_lesion_trajectories, 
            **config.dataset_params
        )
        valid_interpolation_loader = DataLoader(
            valid_interpolation_dataset, 
            shuffle=False
        )

        valid_extrapolation_dataset = Validation_Extrapolation_LesionSequenceDataset(
            mri_dataloader.cache_lesion_trajectories, 
            **config.dataset_params
        )
        valid_extrapolation_loader = DataLoader(
            valid_extrapolation_dataset, 
            shuffle=False
        )

        plotting_LesionDataset = Plotting_LesionSequenceDataset(
            mri_dataloader.cache_lesion_trajectories.copy(), 
            **config.dataset_params
        )

        model = config.model(
            trajectories=trajectories, 
            **config.model_params
        )

        
        mlflow.log_params({
            "dataset_size": len(train_dataset),
            "loss_function": type(config.loss_fn).__name__,
            **{key: getattr(config, key) for key in dir(config) if not key.startswith("_")}
        })
        mlflow.log_artifact(args.config)
        mlflow.log_artifact(os.path.basename(__file__))

        losses = train_lstm(
            model, 
            train_loader, 
            valid_interpolation_loader, 
            valid_extrapolation_loader, 
            plotting_LesionDataset,
            config
        )
        if len(losses) != 0:
            mlflow.log_metric("final_train_loss", losses[-1])