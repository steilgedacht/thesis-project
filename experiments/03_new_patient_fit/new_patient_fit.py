import os
os.chdir("/home/benjaminb/Dokumente/JKU/Semester_9/Practical_Work")

import mlflow
import torch
import json
import argparse
import matplotlib
import importlib.util
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mlflow.pytorch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader import LesionDataset, Plotting_LesionDataset
from utils.train_plotting import plot_lesion_time_evolution, plot_contanct_sheet
from utils.loss_bce_dice import Loss_BCE_Dice
from umap import UMAP

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lesion_INR_Training")

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



def validate_polation(data_loader, text, global_step, criterion):
    loss = 0.0
    volume_error_cm3 = 0.0

    batch_size = getattr(data_loader, 'batch_size', 1) or 1

    for batch_idx, (v_coords, v_labels, v_p_idx) in enumerate(tqdm(data_loader, desc=text, total=len(data_loader))):
        v_coords, v_labels, v_p_idx = v_coords.to(config.device), v_labels.unsqueeze(-1).to(config.device), v_p_idx.to(config.device)
        v_p_idx = torch.zeros_like(v_p_idx)
        v_preds = model(v_coords, torch.zeros_like(v_p_idx, device="cuda"))
        l = criterion.dice_loss(v_preds, v_labels).item()
        loss += l
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



def visualize_embedding_change(model, patient_idx_tensor, train_loader):
    """Visualize embedding before/after training using UMAP projection"""
    
    # Get all embeddings from the model to fit UMAP
    all_embeddings = model.latent_vectors.weight.detach().cpu().numpy()
    
    # Fit UMAP on all embeddings
    umap = UMAP(n_components=2, random_state=42)
    all_embeddings_umap = umap.fit_transform(all_embeddings)
    
    # Get the specific patient's embedding (already fine-tuned)
    final_embedding = model.latent_vectors(patient_idx_tensor).detach().cpu().numpy()
    final_umap = umap.transform(final_embedding)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot all embeddings
    ax1.scatter(all_embeddings_umap[:, 0], all_embeddings_umap[:, 1], alpha=0.5, s=30, label='Other patients')
    ax1.scatter(final_umap[:, 0], final_umap[:, 1], color='red', s=200, marker='*', label='New patient (after)', edgecolors='black', linewidth=2)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Embedding in UMAP Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot embedding dimensions heatmap
    fig2, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    # Show the full 128-d embedding as heatmap
    axes[0].imshow(final_embedding.reshape(1, -1), cmap='coolwarm', aspect='auto')
    axes[0].set_title('Fine-tuned Embedding (128 dimensions)')
    axes[0].set_ylabel('Patient')
    axes[0].set_xlabel('Embedding Dimension')
    
    # Show variance across all embeddings
    embedding_std = np.std(all_embeddings, axis=0)
    axes[1].plot(embedding_std)
    axes[1].set_title('Std Dev Across All Embeddings (shows which dims vary most)')
    axes[1].set_xlabel('Embedding Dimension')
    axes[1].set_ylabel('Std Dev')
    
    mlflow.log_figure(fig, "embedding_pca_projection.png")
    mlflow.log_figure(fig2, "embedding_heatmap.png")
    plt.close(fig)
    plt.close(fig2)


def fit_trajectory_to_model(model, train_loader, plotting_LesionDataset, config, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.AdamW(model.latent_vectors.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25)    
    criterion = Loss_BCE_Dice()

    model.to(device)
    losses = []

    embedding_history = [model.latent_vectors.weight[0].clone().detach().cpu().numpy()]

    # freeze all layers except the embedding layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.latent_vectors.parameters():
        param.requires_grad = True
    
    global_step = epoch = 0

    for epoch in range(epochs):
        total_loss = last_loss = 0
        model.train()
        for i, (coords, labels, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):

            coords = coords.to(device)
            labels = labels.to(device).unsqueeze(-1)
            patient_idx = torch.tensor([0], device="cuda")

            if coords.shape[1] == 0:
                continue

            optimizer.zero_grad()

            predictions = model(coords, patient_idx)

            loss = criterion(predictions, labels)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
            tr_loss = loss.item()

            mlflow.log_metrics(
                {
                    "training_loss" : loss.item(),
                    "training_loss_bce" : criterion.loss_bce.item(),
                    "training_loss_dice" : criterion.loss_dice.item(),
                    "number_of_correctly_predicted_1_labels" : (predictions>0.5).sum().item() / (labels > 0.5).sum().item(),
                },
                step=global_step
            )

            global_step += 1

            if i % 10 == 0:
                del coords, labels, patient_idx, predictions, loss
                torch.cuda.empty_cache()

            current_embedding = model.latent_vectors.weight[0].detach().cpu().numpy()
            embedding_history.append(current_embedding)

            scheduler.step(metrics=tr_loss)

            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=global_step)

        model.eval()

        torch.cuda.empty_cache()

        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    try:
        a = model.latent_vectors.weight[0].detach().cpu().numpy()
        np.save("vec.npz", a)
    except:
        pass

    with torch.no_grad():
        plotting_LesionDataset.trajectories = [mri_dataloader.find_lesion_trajectory('YG_WCONBI42RLEZ', 1)]
        trj = next(iter(plotting_LesionDataset))
        trj.embedding_id = np.int64(0)
        validate_polation(train_loader, "Valid Interpolation", global_step, criterion)
        plot_contanct_sheet(model, epoch, trj, config, final_side_length=True)
        plot_lesion_time_evolution(model, epoch, trj, config, final_side_length=True)


    with torch.no_grad():
        embedding_history = np.array(embedding_history)
        umap = UMAP(n_components=2, random_state=42)
        trajectory_umap = umap.fit_transform(embedding_history)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(trajectory_umap[:, 0], trajectory_umap[:, 1], 'o-', markersize=5)
        ax.scatter(trajectory_umap[0, 0], trajectory_umap[0, 1], color='green', s=200, marker='o', label='Start')
        ax.scatter(trajectory_umap[-1, 0], trajectory_umap[-1, 1], color='red', s=200, marker='*', label='End')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Embedding Trajectory During Fine-tuning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "embedding_trajectory.png")
        plt.close(fig)
        visualize_embedding_change(model, torch.tensor([0]).to(device), train_loader)
        

    return losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with mlflow.start_run(run_name="embedding learning", tags={"type" : "new_patient_fit"}):
    argument_parser = argparse.ArgumentParser(description="Train a Lesion Trajectory model.")
    argument_parser.add_argument("--config", type=str, default="configs/00_default/config.py", help="Path to the configuration file.")
    args = argument_parser.parse_args()
    config = load_config(args.config)
    config = add_base_configurations(config)

    mri_dataloader = MRI_Dataloader()
    trj = [mri_dataloader.find_lesion_trajectory('YG_WCONBI42RLEZ', 1)] * 20
    
    train_dataset = LesionDataset(trj, device=device)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=7, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    plotting_LesionDataset = Plotting_LesionDataset(trj)

    run_id = "8fe747bb5b064f70be27b66dcb418617"
    model_uri = f"runs:/{run_id}/lesion_inr_model"
    model = mlflow.pytorch.load_model(model_uri)

    losses = fit_trajectory_to_model(model, train_loader, plotting_LesionDataset, config=config,  epochs=100, lr=100., device=device)
