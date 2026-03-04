import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mri_dataloader import MRI_Dataloader, Patient
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lesion_INR_Training")

class ModulatedSirenLayer(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        # FiLM generator: projects latent vector to scale (gamma) and shift (beta)
        # We use out_features * 2 because we need a pair for every neuron in this layer
        self.conditioning_lin = nn.Linear(latent_dim, out_features) 
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x, latent):
        # x shape: [batch, num_samples, hidden_dim]
        # latent shape: [batch, latent_dim]
        
        # Generate modulation parameter from latent vector
        # Unsqueeze to align with spatial samples: [batch, 1, hidden_dim]
        modulation = self.conditioning_lin(latent).unsqueeze(1)
        
        # Apply SIREN logic with FiLM modulation
        # We modulate the pre-activation features
        return torch.sin(self.omega_0 * (self.linear(x) + modulation))

class LesionINR(nn.Module):
    def __init__(self, numpatients, latent_dim=128, input_dim=4, hidden_dim=512, output_dim=1, omega_0=30.0):
        super().__init__()
        self.latent_vectors = nn.Embedding(numpatients, latent_dim)
        
        # First layer (takes coordinates)
        self.first_layer = ModulatedSirenLayer(input_dim, hidden_dim, latent_dim, is_first=True, omega_0=omega_0)
        
        # Hidden layers
        self.layers = nn.ModuleList([
            ModulatedSirenLayer(hidden_dim, hidden_dim, latent_dim, is_first=False, omega_0=omega_0)
            for _ in range(4)
        ])
        
        # Final layer (no modulation needed usually, just maps to output)
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.omega_0 = omega_0
        
        # Proper init for latent
        torch.nn.init.normal_(self.latent_vectors.weight, std=1.0 / np.sqrt(latent_dim))

    def forward(self, x, patient_idx):
        # Get latent vector for the patient
        z = self.latent_vectors(patient_idx) # [batch, latent_dim]
        
        # Pass through layers, injecting z at every step
        x = self.first_layer(x, z)
        for layer in self.layers:
            x = layer(x, z)
            
        return self.final_layer(x)

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

class LesionDataset(Dataset):
    def __init__(self, trajectories, device='cuda', context_radius=5, background_samples_proportion=1, mode='train', val_date_idx=None, val_end_date_idx=None):
        self.device = device
        self.trajectories = trajectories
        self.shape = (500, 500, 50)
        self.meshgrid = np.meshgrid(np.linspace(0, 1, self.shape[0]),
                              np.linspace(0, 1, self.shape[1]),
                              np.linspace(0, 1, self.shape[2]),
                              indexing='ij')
        self.context_radius = context_radius
        self.background_samples_proportion = background_samples_proportion
        self.patient_to_idx = {p.patient_id: i for i, p in enumerate(trajectories)}
        self.mode = mode

        if val_date_idx is None:
            self.val_date_idx = [np.random.choice(trj.dates[1:-1]) for trj in trajectories]
        else:
            self.val_date_idx = val_date_idx

        if val_end_date_idx is None:
            self.val_end_date_idx = np.random.choice(range(len(trajectories)), size=len(trajectories) // 10, replace=False)
        else:
            self.val_end_date_idx = val_end_date_idx
        
        if self.mode == 'valid_extrapolation':
            self.trajectories = [trj for i, trj in enumerate(trajectories) if i in self.val_end_date_idx]

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        patient_idx = self.patient_to_idx[trj.patient_id]

        if self.mode == 'train':
            if idx in self.val_end_date_idx:
                random_time_point = str(np.random.choice(trj.dates[:-1]))            
            else:
                random_time_point = str(np.random.choice(trj.dates))
        else:
            if self.mode == 'valid_extrapolation':
                random_time_point = str(trj.dates[-1])
            else:
                random_time_point = str(self.val_date_idx[idx])

        try:
            labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point)
        except:
            patient = Patient(trj.patient_id)
            patient.merge_lesion_to_trajectory()
            try:
                labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point)
            except:
                print(trj.patient_id, random_time_point)
                raise 

        # interpoltate the labels to the shape
        original_shape = labels.shape
        zoom_factors = (
            self.shape[0] / original_shape[0],
            self.shape[1] / original_shape[1],
            self.shape[2] / original_shape[2]
        )
        
        labels = zoom(labels, zoom_factors, order=1)
        x, y, z = np.copy(self.meshgrid)
        

        # 1. Labels binarisieren für schnellere Logik
        lesion_mask = labels > 0.5
        
        # 2. Koordinaten der positiven Voxel (Läsion)
        pos_coords = np.argwhere(lesion_mask)
        
        # 3. Negative Voxel (Hintergrund) finden
        # Trick: Statt argwhere auf dem ganzen Bild, sample einfach zufällige Punkte
        # und schaue, ob sie NICHT in der Maske liegen.
        MAX_SAMPLES = 1000
        num_neg_needed = MAX_SAMPLES // 2
        neg_coords = []
        while len(neg_coords) < num_neg_needed:
            candidate_coords = np.array([
                np.random.randint(0, s, num_neg_needed) for s in self.shape
            ]).T
            # Prüfe welche Kandidaten Hintergrund sind
            is_bg = labels[candidate_coords[:,0], candidate_coords[:,1], candidate_coords[:,2]] <= 0.5
            neg_coords.extend(candidate_coords[is_bg])
        
        neg_coords = np.array(neg_coords)[:num_neg_needed]
        
        # 4. Positive samples (mit Replacement falls Läsion klein ist)
        if len(pos_coords) == 0:
            all_sampled_indices = np.concatenate([neg_coords, neg_coords], axis=0)
        else:
            pos_idx = np.random.choice(len(pos_coords), size=MAX_SAMPLES//2, replace=True)
            pos_sampled = pos_coords[pos_idx]
            
            # 5. Zusammenführen
            all_sampled_indices = np.vstack([pos_sampled, neg_coords])
        
        # 6. Vektorisierte Extraktion aus meshgrid (Kein Loop!)
        i, j, k = all_sampled_indices.T
        coords = np.stack([
            self.meshgrid[0][i, j, k] * 2 - 1,
            self.meshgrid[1][i, j, k] * 2 - 1,
            self.meshgrid[2][i, j, k] * 2 - 1,
            np.full(len(i), time_point)
        ], axis=1)
        
        labels_sampled = labels[i, j, k]

        return coords.astype(np.float32), labels_sampled.astype(np.float32), patient_idx        

def plot_predictions(v_preds, v_labels, v_coords, epoch, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Get 3D prediction volume by reshaping predictions
    pred_volume = v_preds[0].squeeze().cpu().numpy()
    label_volume = v_labels[0].squeeze().cpu().numpy()
    pred_coords = v_coords[0,:,:3].squeeze().cpu().numpy()

    # Top view (XY plane, max projection along Z)
    axes[1,0].scatter(pred_coords[:,0], pred_coords[:,1], c=pred_volume, s=5, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
    axes[1,0].set_title('Prediction - Top View')
    axes[1,0].axis('off')

    # Side view (XZ plane, max projection along Y)
    axes[0,0].scatter(pred_coords[:,0], pred_coords[:,2], c=pred_volume, s=5, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
    axes[0,0].set_title('Prediction - Front View')
    axes[0,0].axis('off')

    axes[0,1].scatter(pred_coords[:,1], pred_coords[:,2], c=pred_volume, s=5, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
    axes[0,1].set_title('Prediction - Side View')
    axes[0,1].axis('off')

    axes[1,1].scatter(pred_coords[:,0], pred_coords[:,1], c=label_volume, s=5, alpha=0.5, cmap='viridis', vmin=0, vmax=1)
    axes[1,1].set_title('Labels')
    axes[1,1].axis('off')


    plt.tight_layout()
    mlflow.log_figure(fig, f"{epoch}_epoch_predictions_{title}.png")
    plt.close(fig)

def train_inr(model, train_loader, valid_interpolation_loader, valid_extrapolation_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)    
    criterion = nn.BCEWithLogitsLoss()

    model.to(device)
    losses = []
    
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("hidden_dim", model.layers[0].linear.out_features)
    mlflow.log_param("omega_0", model.omega_0)
    mlflow.log_param("batch_size", train_loader.batch_size)
    
    global_step = 0

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

            loss_bce = criterion(predictions, labels)
            loss_dice = dice_loss(predictions, labels)
            loss = loss_bce + loss_dice
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            total_loss += loss.item()

            mlflow.log_metric("training_loss", loss.item(), step=global_step)
            mlflow.log_metric("training_loss_bce", loss_bce.item(), step=global_step)
            mlflow.log_metric("training_loss_dice", loss_dice.item(), step=global_step)
            mlflow.log_metric("number_of_correctly_predicted_1_labels", (predictions>0.5).sum().item() / (labels > 0.5).sum().item(), step=global_step)
            mlflow.log_metric("number_of_correctly_predicted_0_labels", (predictions<=0.5).sum().item() / (labels <= 0.5).sum().item(), step=global_step)
            global_step += 1

            if i % 10 == 0:
                del coords, labels, patient_idx, predictions, loss
                torch.cuda.empty_cache()
        
        plot_predictions(predictions, labels, coords, epoch, "train")

        

        model.eval()
        total_val_inter = 0
        with torch.no_grad():
            for v_coords, v_labels, v_p_idx in valid_interpolation_loader:
                v_coords, v_labels, v_p_idx = v_coords.to(device), v_labels.unsqueeze(-1).to(device), v_p_idx.to(device)
                v_preds = model(v_coords, v_p_idx)
                total_val_inter += (1 - dice_loss(v_preds, v_labels)).item()

        total_val_extrap = 0
        with torch.no_grad():
            for v_coords, v_labels, v_p_idx in valid_extrapolation_loader:
                v_coords, v_labels, v_p_idx = v_coords.to(device), v_labels.unsqueeze(-1).to(device), v_p_idx.to(device)
                v_preds = model(v_coords, v_p_idx)
                total_val_extrap += (1 - dice_loss(v_preds, v_labels)).item()
        
        with torch.no_grad():
            plot_predictions(v_preds, v_labels, v_coords, epoch, "validation")

        mlflow.log_metric("validation_interpolation", total_val_inter / len(valid_interpolation_loader), step=global_step)
        mlflow.log_metric("validation_extrapolation", total_val_extrap / len(valid_extrapolation_loader), step=global_step)
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)

        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)        
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    mlflow.pytorch.log_model(model, name="lesion_inr_model")
    
    return losses

# Trainingscode
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MLflow Run starten
with mlflow.start_run():
    mlflow.log_param("device", device)

    mri_dataloader = MRI_Dataloader()
    mri_dataloader.cache_lesion_trajectories_from_n_scans(10)
    trajectories = mri_dataloader.cache_lesion_trajectories
    
    train_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    
    valid_interpolation_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='valid_interpolation', val_date_idx=train_dataset.val_date_idx, val_end_date_idx=train_dataset.val_end_date_idx)
    valid_interpolation_loader = DataLoader(valid_interpolation_dataset, batch_size=16, shuffle=False)

    valid_extrapolation_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='valid_extrapolation', val_date_idx=train_dataset.val_date_idx, val_end_date_idx=train_dataset.val_end_date_idx)
    valid_extrapolation_loader = DataLoader(valid_extrapolation_dataset, batch_size=16, shuffle=False)

    mlflow.log_param("dataset_size", len(train_dataset))
    
    model = LesionINR(len(trajectories), latent_dim=64, input_dim=4, hidden_dim=2048, output_dim=1)

    losses = train_inr(model, train_loader, valid_interpolation_loader, valid_extrapolation_loader, epochs=40, lr=1e-5, device=device)
    
    final_loss = losses[-1]
    mlflow.log_metric("final_train_loss", final_loss)