"""Neural ODE INR with a separate time-varying lesion translation.

The shape latent describes the lesion in coordinates relative to its predicted
center. A separate three-dimensional ODE state predicts that center's
translation in the same normalized xyz units used by the input coordinates.
"""

import torch
import torch.nn as nn

from .model_neural_ode import (
    FixedStepODESolver,
    FourierFeatures,
    LatentODEFunc,
    SpatialDecoder,
)


class NeuralODETranslationINR(nn.Module):
    """Drop-in NeuralODE_INR variant with an explicit xyz translation state.

    The patient embedding initializes independent shape and translation
    states. The shape state drives the spatial decoder; the translation state
    shifts coordinates before spatial encoding. Both states evolve over time
    with the same ODE solver settings.
    """

    def __init__(
        self,
        trajectories=None,
        num_patients=1,
        patient_embed_dim=32,
        latent_dim=64,
        ode_hidden_dim=128,
        ode_layers=2,
        ode_steps=16,
        ode_method="rk4",
        max_t=3650.0,
        spatial_hidden_dim=256,
        spatial_layers=4,
        num_fourier_frequencies=6,
    ):
        super().__init__()

        self.patient_embedding = nn.Embedding(num_patients, patient_embed_dim)
        self.max_t = max_t

        self.initial_state_net = nn.Sequential(
            nn.Linear(patient_embed_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.shape_ode_solver = FixedStepODESolver(
            LatentODEFunc(
                latent_dim=latent_dim,
                cond_dim=patient_embed_dim,
                hidden_dim=ode_hidden_dim,
                n_layers=ode_layers,
            ),
            n_steps=ode_steps,
            method=ode_method,
        )

        self.initial_translation_net = nn.Linear(patient_embed_dim, 3)
        nn.init.zeros_(self.initial_translation_net.weight)
        nn.init.zeros_(self.initial_translation_net.bias)
        self.translation_ode_solver = FixedStepODESolver(
            LatentODEFunc(
                latent_dim=3,
                cond_dim=patient_embed_dim,
                hidden_dim=ode_hidden_dim,
                n_layers=ode_layers,
            ),
            n_steps=ode_steps,
            method=ode_method,
        )

        self.spatial_encoding = FourierFeatures(
            in_dim=3,
            num_frequencies=num_fourier_frequencies,
        )
        self.decoder = SpatialDecoder(
            spatial_in_dim=self.spatial_encoding.out_dim,
            latent_dim=latent_dim,
            hidden_dim=spatial_hidden_dim,
            n_layers=spatial_layers,
        )

    @staticmethod
    def _shared_time(coords):
        return coords[..., 3].mean(dim=1, keepdim=True)

    def forward(self, coords, patient_idx):
        xyz = coords[..., :3]
        t = self._shared_time(coords) / self.max_t

        patient_emb = self.patient_embedding(patient_idx)

        shape_state = self.initial_state_net(patient_emb)
        shape_state = self.shape_ode_solver(shape_state, t, patient_emb)

        initial_translation = self.initial_translation_net(patient_emb)
        translation = self.translation_ode_solver(
            initial_translation,
            t,
            patient_emb,
        )

        relative_xyz = xyz - translation.unsqueeze(1)
        xyz_encoded = self.spatial_encoding(relative_xyz)
        shape_state = shape_state.unsqueeze(1).expand(-1, xyz.shape[1], -1)
        return self.decoder(xyz_encoded, shape_state)
