import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mri_dataloader import MRI_Dataloader, Patient
from scipy.ndimage import zoom

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
                self.linear.bias.uniform_(-bound if not self.is_first else 1 / self.in_features, 
                                         bound if not self.is_first else 1 / self.in_features)
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class LesionINR(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.layers = nn.ModuleList([
            SirenLayer(input_dim, hidden_dim, is_first=True, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0),
            SirenLayer(hidden_dim, output_dim, is_first=False, omega_0=omega_0),
        ])
        # Final sigmoid for output normalization
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, coords):
        x = coords
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply sigmoid only to the final output
            if i == len(self.layers) - 1:
                x = self.sigmoid(x)
        return x

class LesionDataset(Dataset):
    def __init__(self, trajectories, device='cuda'):
        self.device = device
        self.trajectories = trajectories
        self.shape = (64, 64, 20)
        self.meshgrid = np.meshgrid(np.linspace(0, 1, self.shape[0]),
                              np.linspace(0, 1, self.shape[1]),
                              np.linspace(0, 1, self.shape[2]),
                              indexing='ij')
            
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trj = self.trajectories[idx]
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
        x, y, z = self.meshgrid
        
        coords = np.stack([x, y, z, np.full_like(x, time_point)], axis=-1)
        coords = coords.reshape(-1, 4)
        labels = np.concatenate([label.flatten() for label in labels])
        
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        return coords, labels
    
def train_inr(model, train_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.to(device)
    losses = []
    
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("hidden_dim", model.layers[0].linear.out_features)
    mlflow.log_param("omega_0", model.omega_0)
    mlflow.log_param("batch_size", train_loader.batch_size)
    
    for epoch in range(epochs):
        with mlflow.start_span(f"Epoch {epoch+1}"):
            total_loss = 0
            for coords, labels in train_loader:
                optimizer.zero_grad()
                predictions = model(coords)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                del coords, labels, predictions, loss
                torch.cuda.empty_cache()
            
            avg_loss = total_loss / len(train_loader)
            losses.append(avg_loss)
            
            mlflow.log_metric("loss", avg_loss, step=epoch)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    mlflow.pytorch.log_model(model, "lesion_inr_model")
    
    return losses

# Trainingscode
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# MLflow Run starten
with mlflow.start_run():
    with mlflow.start_span("Experiment"):
        mlflow.log_param("device", device)

        with mlflow.start_span("Data Loading and Preprocessing"):
            mri_dataloader = MRI_Dataloader()
            mri_dataloader.cache_lesion_trajectories_from_n_scans(6)
            
            trajectories = mri_dataloader.cache_lesion_trajectories
            
            dataset = LesionDataset(trajectories, device=device)
            train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
            
            mlflow.log_param("dataset_size", len(dataset))
            
            model = LesionINR(input_dim=4, hidden_dim=128, output_dim=1)
        
        with mlflow.start_span("Model Training"):
            losses = train_inr(model, train_loader, epochs=100, lr=1e-3, device=device)
        
        final_loss = losses[-1]
        mlflow.log_metric("final_loss", final_loss)