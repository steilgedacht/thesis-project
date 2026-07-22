import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mri_dataloader import MRI_Dataloader
from tqdm import tqdm
from utils.inr_model import LesionINR
from utils.loss_bce_dice import Loss_BCE_Dice
from utils.dataloader import LesionDataset
from utils.train_plotting import *

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lesion_INR_Training")


def validate_polation(data_loader, text, global_step, criterion):
    loss = 0
    for v_coords, v_labels, v_p_idx in tqdm(data_loader, desc=text, total=len(data_loader)):
        v_coords, v_labels, v_p_idx = v_coords.to(device), v_labels.unsqueeze(-1).to(device), v_p_idx.to(device)
        v_preds = model(v_coords, v_p_idx)
        loss += criterion.dice_loss(v_preds, v_labels).item()
    mlflow.log_metric(text.lower().replace(" ", "_"), loss / len(data_loader), step=global_step)


def train_inr(model, train_loader, valid_interpolation_loader, valid_extrapolation_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.0001)    
    criterion = Loss_BCE_Dice()

    model.to(device)
    losses = []
    
    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": model.layers[0].linear.out_features,
        "omega_0": model.omega_0,
        "batch_size": train_loader.batch_size
    })
    
    global_step = 0

    monitoring_samples = np.random.choice(len(train_loader), size=5, replace=False)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):

            coords = coords.to(device)
            labels = labels.to(device).unsqueeze(-1)
            patient_idx = patient_idx.to(device)

            if coords.shape[1] == 0:
                continue

            optimizer.zero_grad()

            predictions = model(coords, patient_idx)

            loss = criterion(predictions, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            total_loss += loss.item()

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

            if i % 10 == 0:
                del coords, labels, patient_idx, predictions, loss
                torch.cuda.empty_cache()

        model.eval()
        torch.cuda.empty_cache()

        if epoch % 20 == 0 and epoch != 0:
            with torch.no_grad():
                visualize_samples(model, epoch, monitoring_samples, train_loader, "train")
                visualize_samples(model, epoch, monitoring_samples, valid_interpolation_loader, "valid")
                validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion)
                validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion)
                for m in monitoring_samples:
                    plot_lesion_time_evolution(model, epoch, m, train_loader)

        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    mlflow.pytorch.log_model(
        model, 
        name="lesion_inr_model", 
        serialization_format="pickle"
    )
        
    for m in monitoring_samples:
        plot_lesion_time_evolution(model, epoch, m, train_loader, side_length=500)

    return losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with mlflow.start_run():
    mri_dataloader = MRI_Dataloader()
    mri_dataloader.cache_lesion_trajectories_from_n_scans(n_scans=6, only_growing=True)
    trajectories = mri_dataloader.cache_lesion_trajectories * 20

    batchsize = 100
    epochs = 200
    background_samples = 3000
    
    train_dataset = LesionDataset(
        trajectories, 
        context_radius=5, 
        background_samples_proportion=1, 
        device=device, 
        mode='train', 
        background_samples=background_samples
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batchsize, 
        shuffle=True, 
        num_workers=7, 
        prefetch_factor=2, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    valid_interpolation_dataset = LesionDataset(
        trajectories, 
        context_radius=5, 
        background_samples_proportion=1, 
        device=device, 
        mode='valid_interpolation', 
        val_date_idx=train_dataset.val_date_idx, 
        val_end_date_idx=train_dataset.val_end_date_idx, 
        background_samples=background_samples
    )
    valid_interpolation_loader = DataLoader(
        valid_interpolation_dataset, 
        batch_size=batchsize, 
        shuffle=False
    )

    valid_extrapolation_dataset = LesionDataset(
        trajectories, 
        context_radius=5, 
        background_samples_proportion=1, 
        device=device, 
        mode='valid_extrapolation', 
        val_date_idx=train_dataset.val_date_idx, 
        val_end_date_idx=train_dataset.val_end_date_idx, 
        background_samples=background_samples
    )
    valid_extrapolation_loader = DataLoader(
        valid_extrapolation_dataset, 
        batch_size=batchsize, 
        shuffle=False
    )

    model = LesionINR(
        len(trajectories)*100, 
        latent_dim=128, 
        input_dim=4, 
        hidden_dim=512, 
        output_dim=1, 
        n_layers=8
    )
    mlflow.log_params({
        "dataset_size": len(train_dataset), 
        "batch_size": batchsize,
        "latent_dim": model.latent_dim,
        "hidden_dim": model.hidden_dim,
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "n_layers": model.n_layers,
        "omega" : model.omega_0,
        "device": device,
        "epochs": epochs,
        "background_samples": background_samples
    })
    mlflow.log_artifact("inr.py")

    losses = train_inr(
        model, 
        train_loader, 
        valid_interpolation_loader, 
        valid_extrapolation_loader, 
        epochs=epochs, 
        lr=1e-4, 
        device=device
    )
    
    final_loss = losses[-1]
    mlflow.log_metric("final_train_loss", final_loss)