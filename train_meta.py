import importlib.util
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import argparse
import requests
import json
import higher  # <-- Required for Meta-Learning inner loops
import os

# Utility imports (Assuming these are in your local directory)
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader import LesionDataset, Validation_Extrapolation_LesionDataset, Validation_Interpolation_LesionDataset, Plotting_LesionDataset
from utils.train_plotting_meta import *

def check_if_mlflow_is_running(config):
    try:
        requests.get(f"{config.mlflow_tracking_uri}/version")
    except Exception as e:
        import shutil
        import time
        shutil.os.system("tmux new-session -d -s mlflow_server 'mlflow server'")
        print("Starting Mlflow in a new tmux session...")
        time.sleep(5)

def load_config(config_path: str):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()

def validate_polation(data_loader, text, global_step, criterion, model, config):
    """
    Test-time adaptation validation: adapt to support set, evaluate on query set.
    Uses the same loss function as training for consistency.
    """
    bce_loss_total = 0.0
    dice_loss_total = 0.0
    tv_loss_total = 0.0
    num_batches = 0
    
    model.eval()
    
    # Meta-Learning Hyperparameters for Test-Time Adaptation
    inner_lr = getattr(config, 'inner_lr', 0.01)
    inner_steps = getattr(config, 'inner_steps', 3)
    
    for v_coords, v_labels, v_p_idx in tqdm(data_loader, desc=text, total=len(data_loader)):
        v_coords = v_coords.to(config.device)
        v_labels = v_labels.unsqueeze(-1).to(config.device)
        # v_p_idx is ignored; patient identity is learned via adaptation

        # Split into Support (for test-time adaptation) and Query (for evaluation)
        split_idx = v_coords.shape[0] // 2
        supp_coords, query_coords = v_coords[:split_idx], v_coords[split_idx:]
        supp_labels, query_labels = v_labels[:split_idx], v_labels[split_idx:]

        if supp_coords.shape[0] == 0 or query_coords.shape[0] == 0:
            continue

        inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
        
        # Test-time adaptation (track_higher_grads=False saves memory since we don't update meta-weights here)
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            # Adapt to the patient's support scans using the full loss (BCE + Dice + TV)
            for _ in range(inner_steps):
                supp_preds = fmodel(supp_coords)
                inner_loss = criterion(supp_preds, supp_labels)
                diffopt.step(inner_loss)

            # Evaluate on the patient's query scans
            with torch.no_grad():
                query_preds = fmodel(query_coords)
                # Use full loss for evaluation consistency with training
                eval_loss = criterion(query_preds, query_labels)
                
                # Track individual components
                bce_loss_total += criterion.loss_bce.item() if isinstance(criterion.loss_bce, torch.Tensor) else criterion.loss_bce
                dice_loss_total += criterion.loss_dice.item() if isinstance(criterion.loss_dice, torch.Tensor) else criterion.loss_dice
                tv_loss_total += (criterion.loss_tv.item() if isinstance(criterion.loss_tv, torch.Tensor) else criterion.loss_tv) if hasattr(criterion, 'loss_tv') else 0.0
                num_batches += 1

    # Calculate averages
    num_batches = max(num_batches, 1)
    avg_bce = bce_loss_total / num_batches
    avg_dice = dice_loss_total / num_batches
    avg_tv = tv_loss_total / num_batches
    avg_total = avg_bce + avg_dice + avg_tv
    
    mlflow.log_metrics({
        text.lower().replace(" ", "_") + "_bce_loss"  : avg_bce,
        text.lower().replace(" ", "_") + "_dice_loss" : avg_dice,
        text.lower().replace(" ", "_") + "_dice_score": 1 - avg_dice,
    }, step=global_step)
    
    # Log TV loss if enabled
    if hasattr(criterion, 'use_tv_loss') and criterion.use_tv_loss:
        mlflow.log_metrics({
            text.lower().replace(" ", "_") + "_tv_loss"   : avg_tv,
            text.lower().replace(" ", "_") + "_total_loss": avg_total,
        }, step=global_step)
    
    # Log dynamic pos_weight if enabled
    if hasattr(criterion, 'use_dynamic_pos_weight') and criterion.use_dynamic_pos_weight:
        mlflow.log_metrics({
            text.lower().replace(" ", "_") + "_dynamic_pos_weight": criterion.current_pos_weight,
        }, step=global_step)


def train_inr(
        model, 
        train_loader, 
        valid_interpolation_loader, 
        valid_extrapolation_loader,
        plotting_LesionDataset, 
        config,
    ):
    """
    Meta-Learning training loop for INR using MAML-style approach.
    
    Architecture:
    - INNER LOOP (Patient-specific adaptation):
      Adapts model parameters to patient support set using SGD.
    
    - OUTER LOOP (Meta-weight update):
      Updates meta-weights (theta_meta) based on performance on query set,
      enabling rapid adaptation to new patients.
    
    Loss Function:
    - Uses full Loss_BCE_Dice with dynamic pos_weight and TV regularization
    - Inner loop: Adapts on support set losses
    - Outer loop: Computes meta-gradient on query set losses
    - Both use same loss function for consistency
    
    Uses `higher` library for differentiable inner loop computation.
    """
    # Outer optimizer (updates theta_meta)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.lr, 
        weight_decay=config.weight_decay
    )
    
    # Meta-Learning specific config fallbacks
    inner_lr = getattr(config, 'inner_lr', 0.01)
    inner_steps = getattr(config, 'inner_steps', 3)
    inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.epochs, 
        eta_min=config.scheduler_eta_min
    )
    criterion = config.loss_fn

    model.to(config.device)
    losses = []
    global_step = epoch = 0

    for epoch in range(1, config.epochs + 1):
        total_loss = 0

        # ==== Meta-Training Loop ====
        model.train()
        for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}"):

            coords = coords.to(config.device).requires_grad_(True)
            labels = labels.to(config.device).unsqueeze(-1)
            if coords.shape[1] == 0: continue

            # SPLIT BATCH: Support Set (Inner Loop) & Query Set (Outer Loop)
            split_idx = coords.shape[0] // 2
            supp_coords, query_coords = coords[:split_idx], coords[split_idx:]
            supp_labels, query_labels = labels[:split_idx], labels[split_idx:]

            optimizer.zero_grad()

            # higher library context manager for the differentiable inner loop
            with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                
                # --- INNER LOOP (Patient Adaptation) ---
                for _ in range(inner_steps):
                    supp_preds = fmodel(supp_coords)
                    # Use full loss (BCE + Dice + TV) for inner loop adaptation
                    inner_loss = criterion(supp_preds, supp_labels)
                    diffopt.step(inner_loss)

                # --- OUTER LOOP (Meta-Weight Update) ---
                query_preds = fmodel(query_coords)
                # Use full loss (BCE + Dice + TV) for meta-weight update
                outer_loss = criterion(query_preds, query_labels)

                if outer_loss.requires_grad:
                    outer_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm_clip) 
                    optimizer.step()
            
            total_loss += outer_loss.item()

            if i % config.train_log_interval == 0:
                mlflow.log_metrics(
                    {
                        "training_loss" : outer_loss.item(),
                        **criterion.loss_to_report,  # Includes BCE, Dice, TV, pos_weight components
                        # Approximation based on the query set predictions
                        "wrongly_predicted_total_volume" : (query_preds > 0.5).sum().item() - (query_labels > 0.5).sum().item(),
                    },
                    step=global_step
                )

            global_step += 1

            if i % config.train_delete_cache_interval == 0:
                del coords, labels, patient_idx, supp_preds, query_preds, inner_loss, outer_loss
                torch.cuda.empty_cache()

        if config.only_train: continue

        # ==== Evaluation Loop ====
        if epoch % config.validation_interval == 0:
            # We don't use torch.no_grad() globally here because validate_polation needs gradients for the inner adaptation loop
            validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion, model, config)
            validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion, model, config)
            
            # NOTE: Your plotting function needs to adapt the model first, or it will plot the raw (unadapted) theta_meta.
            # You may need to update `plot_lesion_time_evolution` internally to run an inner loop on `trj` before plotting.
            with torch.no_grad():
                for trj in plotting_LesionDataset:
                    plot_lesion_time_evolution(model, epoch, trj, config)

        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % config.print_loss_interval == 0:
            print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.6f}")

    # ==== Final Logging and Visualization ====
    mlflow.pytorch.log_model(
        model, 
        name=config.model_save_name, 
        serialization_format="pickle"
    )

    with torch.no_grad():
        for trj in plotting_LesionDataset:  
            plot_lesion_time_evolution(model, epoch, trj, config, final_side_length=True)

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
        
        train_dataset = LesionDataset(
            trajectories, 
            dialation_iterations=config.dialation_iterations, 
            device=config.device, 
            background_samples=config.background_samples
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batchsize, 
            shuffle=True, 
            num_workers=config.num_workers, 
            prefetch_factor=config.prefetch_factor, 
            pin_memory=config.pin_memory, 
            persistent_workers=config.persistent_workers
        )
        
        valid_interpolation_dataset = Validation_Interpolation_LesionDataset(
            mri_dataloader.cache_lesion_trajectories, 
            dialation_iterations=config.dialation_iterations, 
            device=config.device, 
            background_samples=config.background_samples
        )
        valid_interpolation_loader = DataLoader(
            valid_interpolation_dataset, 
            shuffle=False
        )

        valid_extrapolation_dataset = Validation_Extrapolation_LesionDataset(
            mri_dataloader.cache_lesion_trajectories, 
            dialation_iterations=config.dialation_iterations, 
            device=config.device, 
            background_samples=config.background_samples
        )
        valid_extrapolation_loader = DataLoader(
            valid_extrapolation_dataset, 
            shuffle=False
        )

        plotting_LesionDataset = Plotting_LesionDataset(
            mri_dataloader.cache_lesion_trajectories.copy(), 
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

        losses = train_inr(
            model, 
            train_loader, 
            valid_interpolation_loader, 
            valid_extrapolation_loader, 
            plotting_LesionDataset,
            config
        )
        if len(losses) != 0:
            mlflow.log_metric("final_train_loss", losses[-1])