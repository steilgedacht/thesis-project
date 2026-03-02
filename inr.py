import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mri_dataloader import MRI_Dataloader, Patient
from scipy.ndimage import zoom
from scipy.ndimage import binary_dilation
from tqdm import tqdm

mlflow.set_experiment("Lesion_INR_Training")

class SirenLayer(nn.Module):
    """SIREN layer with sine activation and proper weight initialization."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform initialization in [-1/in_features, 1/in_features]
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layers: uniform initialization based on omega_0
                bound = np.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

                if self.linear.bias is not None:
                    self.linear.bias.uniform_(-bound if not self.is_first else 1 / self.in_features, bound if not self.is_first else 1 / self.in_features)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class LesionINR(nn.Module):
    def __init__(self, numpatients, latent_dim=128, input_dim=4, hidden_dim=512, output_dim=1, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        combined_input_dim = input_dim + latent_dim
        self.layers = nn.ModuleList([
            SirenLayer(combined_input_dim, hidden_dim, is_first=True, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, output_dim, is_first=False, omega_0=omega_0),
        ])
        self.sigmoid = nn.Sigmoid()
    
        self.latent_vectors = nn.Embedding(numpatients, latent_dim)
        torch.nn.init.normal_(self.latent_vectors.weight, std=0.01)

    def forward(self, x, patient_idx):
        z = self.latent_vectors(patient_idx)
        z_expanded = z.unsqueeze(1).expand(-1, x.size(1), -1) 
        x = torch.cat([x, z_expanded], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x

class LesionDataset(Dataset):
    def __init__(self, trajectories, device='cuda', context_radius=5, background_samples_proportion=1):
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

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        patient_idx = self.patient_to_idx[trj.patient_id]
        random_time_point = str(np.random.choice(trj.dates))
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
        
        # find lesion voxels
        lesion_mask = labels > 0.5
        lesion_coords = np.argwhere(lesion_mask)
        
        # Sammle Sample-Indizes
        sampled_indices = []
        
        # 1. Alle Läsions-Voxel
        sampled_indices.extend([tuple(coord) for coord in lesion_coords])
        
        # 2. Context um Läsion (Dilation)
        context_mask = binary_dilation(lesion_mask, iterations=self.context_radius)
        context_mask = context_mask & ~lesion_mask  # Nur Ring um Läsion, nicht Läsion selbst
        context_coords = np.argwhere(context_mask)
        sampled_indices.extend([tuple(coord) for coord in context_coords])

        MAX_SAMPLES = 10_000
        
        # 3. Zufällige Background-Samples
        background_coords = np.argwhere(~context_mask & ~lesion_mask)
        if len(background_coords) > 0:
            background_idx = np.random.choice(len(background_coords), 
                                            #  size=int(len(sampled_indices) * self.background_samples_proportion), 
                                             size=150_000, 
                                             replace=False)
            sampled_indices.extend([tuple(background_coords[i]) for i in background_idx])
        else:
            sampled_indices.extend([tuple(coord) for coord in background_coords])
        
        # Konvertiere zu Koordinaten und Labels
        coords_list = []
        labels_list = []

        # shuffle sampled indices
        np.random.shuffle(sampled_indices)
        
        for idx_tuple in sampled_indices:
            i, j, k = idx_tuple
            coords_list.append([x[i, j, k] * 2 - 1, y[i, j, k] * 2 - 1, z[i, j, k] * 2 - 1, time_point])
            labels_list.append(labels[i, j, k])
        
        coords = np.array(coords_list)
        labels = np.array(labels_list)

        # sample equal number of both classes to have a balanced dataset
        pos_indices = np.where(labels > 0.5)[0]
        neg_indices = np.where(labels <= 0.5)[0]

        if len(pos_indices) == 0:
            # If no positive samples, return all negative samples
            sampled_indices = neg_indices[:MAX_SAMPLES]
        else:
            pos_sampled = np.random.choice(pos_indices, size=MAX_SAMPLES//2, replace=True)
            neg_sampled = np.random.choice(neg_indices, size=MAX_SAMPLES//2, replace=True)
            sampled_indices = np.concatenate([pos_sampled, neg_sampled])
        coords = coords[sampled_indices]
        labels = labels[sampled_indices]
                
        return coords, labels, patient_idx
    
def train_inr(model, train_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    model.to(device)
    model = model.to(torch.float16)
    losses = []
    
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("hidden_dim", model.layers[0].linear.out_features)
    mlflow.log_param("omega_0", model.omega_0)
    mlflow.log_param("batch_size", train_loader.batch_size)
    
    for epoch in range(epochs):
        with mlflow.start_span(f"Epoch {epoch+1}"):
            total_loss = 0
            for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader)):

                coords = coords.to(device).to(torch.float16)
                labels = labels.to(device).to(torch.float16).unsqueeze(-1)
                patient_idx = patient_idx.to(device).long()

                if coords.shape[1] == 0:
                    continue

                optimizer.zero_grad()
                predictions = model(coords, patient_idx)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_loader) + i)

                if i % 10 == 0:
                    del coords, labels, patient_idx, predictions, loss
                    torch.cuda.empty_cache()
            scheduler.step()

            
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            
            mlflow.log_metric("train_loss_after_epoch", avg_loss, step=epoch * len(train_loader))
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    mlflow.pytorch.log_model(model, "lesion_inr_model")
    
    return losses

# Trainingscode
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MLflow Run starten
with mlflow.start_run():
    mlflow.log_param("device", device)

    with mlflow.start_span("Data Loading and Preprocessing"):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(5)
        
        trajectories = mri_dataloader.cache_lesion_trajectories
        
        dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=6)
        
        mlflow.log_param("dataset_size", len(dataset))
        
        model = LesionINR(len(trajectories), input_dim=4, hidden_dim=512, output_dim=1)
    
    with mlflow.start_span("Model Training"):
        losses = train_inr(model, train_loader, epochs=100, lr=1e-3, device=device)
    
    final_loss = losses[-1]
    mlflow.log_metric("final_loss", final_loss)