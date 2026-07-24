import importlib.util
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader import LesionDataset
from utils.train_plotting import *
import argparse


def load_config(config_path: str):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()


def validate_polation(data_loader, text, global_step, criterion):
    loss = 0
    for v_coords, v_labels, v_p_idx in tqdm(data_loader, desc=text, total=len(data_loader)):
        v_coords, v_labels, v_p_idx = v_coords.to(config.device), v_labels.unsqueeze(-1).to(config.device), v_p_idx.to(config.device)
        v_preds = model(v_coords, v_p_idx)
        loss += criterion.dice_loss(v_preds, v_labels).item()
    mlflow.log_metric(text.lower().replace(" ", "_"), loss / len(data_loader), step=global_step)


def train_inr(
        model, 
        train_loader, 
        valid_interpolation_loader, 
        valid_extrapolation_loader, 
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
    
    global_step = 0
    monitoring_samples = np.random.choice(len(train_loader), size=config.n_monitoring_samples_to_visualize, replace=False)

    for epoch in range(1, config.epochs + 1):
        total_loss = 0

        # ==== Training Loop ====
        model.train()
        for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}"):

            coords = coords.to(config.device)
            labels = labels.to(config.device).unsqueeze(-1)
            patient_idx = patient_idx.to(config.device)
            if coords.shape[1] == 0: continue

            optimizer.zero_grad()

            predictions = model(coords, patient_idx)

            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm_clip) 
            optimizer.step()
            total_loss += loss.item()

            if i % config.train_log_interval == 0:
                mlflow.log_metrics(
                    {
                        "training_loss" : loss.item(),
                        "training_loss_bce" : criterion.loss_bce.item(),
                        "training_loss_dice" : criterion.loss_dice.item(),
                        "number_of_correctly_predicted_1_labels" : (predictions>0.5).sum().item() / (labels > 0.5).sum().item(),
                        "number_of_correctly_predicted_0_labels" : (predictions<=0.5).sum().item() / (labels <= 0.5).sum().item()
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
                visualize_samples(model, epoch, monitoring_samples, train_loader, "train", config)
                visualize_samples(model, epoch, monitoring_samples, valid_interpolation_loader, "valid", config)
                validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion)
                validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion)
                for sample in monitoring_samples:
                    plot_lesion_time_evolution(model, epoch, sample, train_loader, config)

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
        
    for sample in monitoring_samples:
        plot_lesion_time_evolution(model, epoch, sample, train_loader, config, final_side_length=True)

    return losses

if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(description="Train a Lesion Trajectory model.")
    argument_parser.add_argument("--config", type=str, default="configs/00_default/config.py", help="Path to the configuration file.")
    config = load_config(argument_parser.parse_args().config)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(
            n_scans=config.lesion_trajectories_with_more_than_n_scans, 
            only_growing=config.use_only_growing_lesions
        )
        if mri_dataloader.cache_lesion_trajectories is not None:
            trajectories = mri_dataloader.cache_lesion_trajectories * config.training_dataset_samples_duplication_factor
        
        train_dataset = LesionDataset(
            trajectories, 
            dialation_iterations=config.dialation_iterations, 
            device=config.device, 
            mode='train', 
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
        
        valid_interpolation_dataset = LesionDataset(
            trajectories, 
            dialation_iterations=config.dialation_iterations, 
            device=config.device, 
            mode='valid_interpolation', 
            val_date_idx=train_dataset.val_date_idx, 
            val_end_date_idx=train_dataset.val_end_date_idx, 
            background_samples=config.background_samples
        )
        valid_interpolation_loader = DataLoader(
            valid_interpolation_dataset, 
            batch_size=config.batchsize, 
            shuffle=False
        )

        valid_extrapolation_dataset = LesionDataset(
            trajectories, 
            dialation_iterations=config.dialation_iterations, 
            device=config.device, 
            mode='valid_extrapolation', 
            val_date_idx=train_dataset.val_date_idx, 
            val_end_date_idx=train_dataset.val_end_date_idx, 
            background_samples=config.background_samples
        )
        valid_extrapolation_loader = DataLoader(
            valid_extrapolation_dataset, 
            batch_size=config.batchsize, 
            shuffle=False
        )

        model = config.model(
            num_patients=len(trajectories), 
            **config.model_params
        )
        
        mlflow.log_params({
            "dataset_size": len(train_dataset),
            "loss_function": type(config.loss_fn).__name__,
            **{key: getattr(config, key) for key in dir(config) if not key.startswith("_")}
        })
        mlflow.log_artifact("config.py")
        mlflow.log_artifact("train.py")

        losses = train_inr(
            model, 
            train_loader, 
            valid_interpolation_loader, 
            valid_extrapolation_loader, 
            config
        )
        
        mlflow.log_metric("final_train_loss", losses[-1])