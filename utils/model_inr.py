import json

import torch
import torch.nn as nn
import numpy as np

from .paths import DatasetPaths
from .region_prior import RegionPrior, apply_region_prior


def lesion_embedding_ids(trajectories):
    """Global embedding ids (patient_idx * 100 + label_id, as LesionDataset.get_embedding_id)
    of the unique lesions in ``trajectories``, plus the size of the global id space."""
    with open(DatasetPaths().patient_to_idx(), "r") as f:
        patient_to_idx = json.load(f)
    ids = sorted({int(patient_to_idx[t.patient_id]) * 100 + int(t.label_id) for t in trajectories})
    id_space = max(max(patient_to_idx.values()) + 1, ids[-1] // 100 + 1) * 100
    return ids, id_space


class TimeEncoder(nn.Module):
    def __init__(self, 
                 max_t, 
                 num_frequencies=6
            ):
        super().__init__()
        self.max_t = max_t
        # This creates a set of frequencies to expand the single time scalar
        self.frequencies = 2**torch.linspace(0, num_frequencies - 1, num_frequencies)

    def forward(self, t):
        # 1. Map to [0, 1]
        t_norm = t / self.max_t 
        
        # 2. Create harmonic features
        # angles shape: [batch, num_frequencies]
        angles = t_norm * self.frequencies.to(t.device) * np.pi
        
        # 3. Concatenate sin and cos: Output dim is 2 * num_frequencies
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embeddings
    
class SirenLayer(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 latent_dim, 
                 is_first=False, 
                 omega_0=30.0
            ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.out_features = out_features
        self.latent_dim = latent_dim
        
        self.linear = nn.Linear(in_features, out_features)
        self.conditioning_lin = nn.Linear(latent_dim, 2 * out_features) 
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
            
            nn.init.zeros_(self.conditioning_lin.weight)
            nn.init.zeros_(self.conditioning_lin.bias)
            self.conditioning_lin.bias.data[:self.out_features] = 1.0

    def forward(self, x, latent):
        modulation = self.conditioning_lin(latent).unsqueeze(1)
        gamma, beta = modulation.chunk(2, dim=-1)
        return torch.sin(self.omega_0 * (gamma * self.linear(x) + beta))

class LesionINR(nn.Module):
    def __init__(self, 
                 trajectories, 
                 latent_dim=128, 
                 input_dim=4, 
                 hidden_dim=512, 
                 output_dim=1, 
                 omega_0=30.0, 
                 n_layers=8,
                 time_freqs=6,
                 max_t=3650.0,
                 use_region_prior=False,
                 region_margin_mm=10.0,
                 region_strength_per_mm=1.0,
                 region_prior_in_training=True,

            ):
        super().__init__()
        # One embedding row per training lesion. Callers keep passing the global
        # embedding id; ``embedding_rows`` maps it to the compact row.
        lesion_ids, id_space = lesion_embedding_ids(trajectories)
        self.num_lesions = len(lesion_ids)
        embedding_rows = torch.zeros(id_space, dtype=torch.long)  # unknown ids -> row 0
        embedding_rows[torch.tensor(lesion_ids)] = torch.arange(len(lesion_ids))
        self.register_buffer("embedding_rows", embedding_rows)
        # Optional far-field prior, indexed by the global id; boxes are filled by
        # utils.region_prior.attach_region_boxes
        self.region_prior = (
            RegionPrior(id_space, region_margin_mm, region_strength_per_mm)
            if use_region_prior else None
        )
        # False: the prior is only applied in eval mode. Applied during training
        # it lets the INR predict "lesion" everywhere near the lesion and leave
        # the shape to the prior (see configs/02_tv_regulation/config_region_prior.py).
        self.region_prior_in_training = bool(region_prior_in_training)
        self.latent_vectors = nn.Embedding(self.num_lesions, latent_dim)
        
        self.time_freqs = time_freqs
        self.time_encoder = TimeEncoder(
            max_t=max_t, 
            num_frequencies=self.time_freqs
        )
        
        # input_dim for latent_adapt is now latent_dim + 1 (for the single normalized time scalar)
        self.latent_adapt = nn.Sequential(
            nn.Linear(latent_dim + (2 * self.time_freqs), latent_dim),
            nn.LeakyReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(),
            nn.LayerNorm(latent_dim)
        )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.omega_0 = omega_0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Initializing layers
        self.first_layer = SirenLayer(3, hidden_dim, latent_dim, is_first=True, omega_0=45)
        self.layers = nn.ModuleList([
            SirenLayer(hidden_dim, hidden_dim, latent_dim, is_first=False, omega_0=omega_0)
            for _ in range(n_layers)
        ])

        self.final_layer = nn.Linear(hidden_dim, output_dim)
        
        with torch.no_grad():
            self.final_layer.weight.uniform_(-np.sqrt(6 / hidden_dim) / omega_0, 
                                             np.sqrt(6 / hidden_dim) / omega_0)
        
        torch.nn.init.normal_(self.latent_vectors.weight, std=1.0 / np.sqrt(latent_dim))

    def forward(self, x, patient_idx):
        spatial_coords = x[..., :3]  
        raw_time = x[:, 0:1, 3] 
        t_encoded = self.time_encoder(raw_time) 
        
        # Checkpoints from before the compact table have no ``embedding_rows``
        rows = getattr(self, "embedding_rows", None)
        z_patient = self.latent_vectors(patient_idx if rows is None else rows[patient_idx])
        z_combined = torch.cat([z_patient, t_encoded], dim=-1)
        
        z = self.latent_adapt(z_combined)
        
        x_out = self.first_layer(spatial_coords, z) 
        for layer in self.layers:
            x_out = layer(x_out, z)

        return apply_region_prior(self, self.final_layer(x_out), spatial_coords, patient_idx)
