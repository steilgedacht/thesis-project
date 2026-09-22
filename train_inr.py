import importlib.util
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader import LesionDataset, Validation_Extrapolation_LesionDataset, Validation_Interpolation_LesionDataset, Plotting_LesionDataset
from utils.train_plotting import *
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
        shutil.os.system("tmux new-session -d -s mlflow_server 'mlflow server'")
        print("Starting Mlflow in a new tmux session...")
        time.sleep(5)

def load_config(config_path: str):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()

def _dataset_item_to_trajectory_and_date(item):
    """Return the trajectory object and the date used for that dataset item."""
    if isinstance(item, tuple) and len(item) == 3:
        _, date, trj = item
        return trj, str(date)
    return item, None


def _voxel_volume_cm3_from_trajectory(trj, selected_date=None):
    """Compute the physical voxel volume for the given trajectory at one date."""
    if selected_date is None:
        selected_date = str(trj.allowed_dates[0] if hasattr(trj, 'allowed_dates') else trj.dates[0])
    _, affine = trj.load_labels_for_inr(selected_date=selected_date, absolute_day_number=True, affine=True)
    voxel_volume_mm3 = float(np.prod(np.abs(np.diag(affine)[:3])))
    return voxel_volume_mm3 / 1000.0

def plot_extrapolation_time(loss_T_pairs, global_step):
    loss_np = np.array([[l[0], l[1].item()] for l in loss_T_pairs])

    x = 180 - loss_np[:, 1]
    y = 1 - loss_np[:, 0]

    # Calculate linear regression slope (m) and intercept (b)
    m, b = np.polyfit(x, y, 1)

    plt.scatter(x, y, label="Extrapolation Scan")
    x_line = np.linspace(1, 180, 200)
    plt.plot(x_line, m * x_line + b, color="red", linestyle="--", label=f"Linear Fit (Slope = {m:.6f})")

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

def validate_polation(data_loader, text, global_step, criterion):
    loss = 0.0
    volume_error_cm3 = 0.0

    batch_size = getattr(data_loader, 'batch_size', 1) or 1

    loss_T_pairs = []

    for batch_idx, (v_coords, v_labels, v_p_idx, T) in enumerate(tqdm(data_loader, desc=text, total=len(data_loader))):
        v_coords, v_labels, v_p_idx = v_coords.to(config.device), v_labels.unsqueeze(-1).to(config.device), v_p_idx.to(config.device)
        v_preds = model(v_coords, v_p_idx)
        l = criterion.dice_loss(v_preds, v_labels).item()
        loss += l 

        loss_T_pairs.append((l, T))

        dataset_start = batch_idx * batch_size
        dataset_end = dataset_start + v_coords.shape[0]
        for local_idx, sample_p_idx in enumerate(v_p_idx.cpu().tolist()):
            data_idx = dataset_start + local_idx
            if data_idx >= len(data_loader.dataset.trajectories):
                break
            dataset_item = data_loader.dataset.trajectories[data_idx]
            trj, date = _dataset_item_to_trajectory_and_date(dataset_item)
            voxel_volume_cm3 = _voxel_volume_cm3_from_trajectory(trj, date)

            pred_count = (torch.sigmoid(v_preds[local_idx]).squeeze(-1) > 0.5).sum().item()
            true_count = (v_labels[local_idx].squeeze(-1) > 0.5).sum().item()
            volume_error_cm3 += abs(pred_count - true_count) * voxel_volume_cm3

    mlflow.log_metrics({
        text.lower().replace(" ", "_") + "_dice_loss"  : loss / len(data_loader),
        text.lower().replace(" ", "_") + "_dice_score" : 1 - loss / len(data_loader),
        text.lower().replace(" ", "_") + "_wrongly_predicted_volume_cm3" : volume_error_cm3,
    }, step=global_step)

    if text == "Valid Extrapolation":
        plot_extrapolation_time(loss_T_pairs, global_step)

def train_inr(
        model, 
        train_loader, 
        valid_interpolation_loader, 
        valid_extrapolation_loader,
        plotting_LesionDataset, 
        config,
    ):
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

    model.to(config.device)
    losses = []
    
    global_step = epoch = 0

    for epoch in range(1, config.epochs + 1):
        total_loss = 0

        # ==== Training Loop ====
        model.train()
        for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{config.epochs}"):

            coords = coords.to(config.device).requires_grad_(True)
            labels = labels.to(config.device).unsqueeze(-1)
            patient_idx = patient_idx.to(config.device)
            if coords.shape[1] == 0: continue

            optimizer.zero_grad()

            predictions = model(coords, patient_idx)

            loss = criterion(predictions, labels, coords=coords if config.use_total_variation_loss else None)

            if loss.requires_grad: # only used for baseline models with no parameters
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm_clip) 
                optimizer.step()
            
            total_loss += loss.item()

            if i % config.train_log_interval == 0:
                mlflow.log_metrics(
                    {
                        "training_loss" : loss.item(),
                        **criterion.loss_to_report,
                        "wrongly_predicted_total_volume" : (predictions > 0.5).sum().item() - (labels > 0.5).sum().item(),
                    },
                    step=global_step
                )

            global_step += 1

            if i % config.train_delete_cache_interval == 0:
                del coords, labels, patient_idx, predictions, loss
                torch.cuda.empty_cache()

        if config.only_train: continue

        # ==== Evaluation Loop ====
        model.eval()

        if epoch % config.validation_interval == 0:
            with torch.no_grad():
                validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion)
                validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion)
                for trj in plotting_LesionDataset:
                    plot_contanct_sheet(model, epoch, trj, config, final_side_length=True)
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
        # if config.load_model:
        #     if config.load_model_path.startswith("runs"):
        #         model = mlflow.pytorch.load_model(config.load_model_path)
        #     else:
        #         model.load_state_dict(torch.load(config.load_model_path, weights_only=True))

        
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