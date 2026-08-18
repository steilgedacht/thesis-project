import torch
import torch.nn as nn
import numpy as np


class TimeEncoder(nn.Module):
    def __init__(self, max_t=3650.0, num_frequencies=6):
        super().__init__()
        self.max_t = max_t
        self.num_frequencies = num_frequencies
        self.register_buffer(
            "frequencies", 
            2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )

    def forward(self, t):
        # Ensure t is at least 2D with a trailing dimension of size 1: shape [..., 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        t_norm = t / self.max_t  # Shape: [..., 1]
        
        # Multiply [..., 1] with [num_frequencies] -> [..., num_frequencies]
        angles = t_norm * self.frequencies * np.pi
        
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embeddings


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.linear.in_features, 
                    1 / self.linear.in_features
                )
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class LesionINR(nn.Module):
    """
    Spatio-Temporal SIREN initialized for MAML / Meta-Learning.
    
    The meta-parameters (theta_meta) capture the common prior across all lesion
    trajectories. In the inner loop, SGD updates these weights directly to adapt
    to a specific patient's support scans.
    """
    def __init__(
        self, 
        trajectories=None,  # Maintained for interface compatibility
        hidden_dim=512, 
        output_dim=1, 
        omega_0=30.0, 
        n_layers=8,
        time_freqs=6,
        max_t=3650.0,
        **kwargs
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(max_t=max_t, num_frequencies=time_freqs)
        
        # Spatial (3) + Fourier Time Features (2 * time_freqs)
        in_dim = 3 + (2 * time_freqs)
        
        self.first_layer = SirenLayer(
            in_features=in_dim, 
            out_features=hidden_dim, 
            is_first=True, 
            omega_0=45.0
        )
        
        self.layers = nn.ModuleList([
            SirenLayer(
                in_features=hidden_dim, 
                out_features=hidden_dim, 
                is_first=False, 
                omega_0=omega_0
            )
            for _ in range(n_layers)
        ])

        self.final_layer = nn.Linear(hidden_dim, output_dim)
        
        with torch.no_grad():
            self.final_layer.weight.uniform_(
                -np.sqrt(6 / hidden_dim) / omega_0, 
                np.sqrt(6 / hidden_dim) / omega_0
            )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [N, 4] containing (x, y, z, t)
        """
        spatial_coords = x[..., :3]     # [..., 3]
        raw_time = x[..., 3:4]         # [..., 1] - keeps trailing dimension size 1
        
        t_encoded = self.time_encoder(raw_time)  # [..., 2 * time_freqs]
        in_features = torch.cat([spatial_coords, t_encoded], dim=-1)

        x_out = self.first_layer(in_features)
        for layer in self.layers:
            x_out = layer(x_out)

        return self.final_layer(x_out)