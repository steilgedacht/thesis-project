import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .model_lstm import Decoder3D, Encoder3D, TimeEncoder


class MambaBlock(nn.Module):
    """Mamba-style block with an input-selective state-space scan."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0,
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .repeat(self.d_inner, 1)
        )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)
        dt_initial = torch.exp(
            torch.rand(self.d_inner) * (math.log(1.0) - math.log(0.001))
            + math.log(0.001)
        ).clamp_min(1e-4)
        inverse_softplus = dt_initial + torch.log(-torch.expm1(-dt_initial))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.shape
        residual = inputs
        normalized = self.norm(inputs)
        x, gate = self.in_proj(normalized).chunk(2, dim=-1)
        x = self.conv1d(
            F.pad(x.transpose(1, 2), (self.d_conv - 1, 0))
        )[..., :sequence_length].transpose(1, 2)
        x = F.silu(x)

        projected = self.x_proj(x)
        dt_low_rank, b_values, c_values = torch.split(
            projected, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_low_rank))
        state_transition = -torch.exp(self.A_log.float()).to(dtype=x.dtype)
        state = x.new_zeros(batch_size, self.d_inner, self.d_state)
        outputs = []

        for step in range(sequence_length):
            delta_step = delta[:, step].unsqueeze(-1)
            b_step = b_values[:, step].unsqueeze(1)
            c_step = c_values[:, step].unsqueeze(1)
            decay = torch.exp(delta_step * state_transition.unsqueeze(0))
            state = decay * state + delta_step * b_step * x[:, step].unsqueeze(-1)
            output = (state * c_step).sum(dim=-1)
            output = output + self.D * x[:, step]
            outputs.append(output)

        scanned = torch.stack(outputs, dim=1)
        return residual + self.out_proj(scanned * F.silu(gate))


class LesionLatentMamba(nn.Module):
    """Encode lesion grids and forecast their latent sequence with Mamba blocks."""

    def __init__(
        self,
        trajectories,
        latent_dim=256,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=2,
        time_freqs=6,
        max_t=3650.0,
        encoder_params=None,
        decoder_params=None,
        use_patient_embedding=True,
        grid_size=(64, 64, 64),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_patient_embedding = use_patient_embedding

        encoder_params = dict(encoder_params or {})
        encoder_params.setdefault("grid_size", grid_size)
        decoder_params = dict(decoder_params or {})
        decoder_params.setdefault("out_size", grid_size)
        self.encoder = Encoder3D(latent_dim=latent_dim, **encoder_params)
        self.decoder = Decoder3D(latent_dim=latent_dim, **decoder_params)

        if use_patient_embedding:
            self.patient_emb = nn.Embedding(1362 * 100, latent_dim)
            nn.init.normal_(self.patient_emb.weight, std=1.0 / np.sqrt(latent_dim))

        self.time_encoder = TimeEncoder(max_t=max_t, num_frequencies=time_freqs)
        input_dim = latent_dim + 2 * time_freqs
        if use_patient_embedding:
            input_dim += latent_dim
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    latent_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(latent_dim)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def freeze_autoencoder(self):
        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.decoder, False)

    def unfreeze_autoencoder(self):
        self._set_requires_grad(self.encoder, True)
        self._set_requires_grad(self.decoder, True)

    def freeze_temporal(self):
        self._set_requires_grad(self.input_proj, False)
        self._set_requires_grad(self.mamba_layers, False)
        self._set_requires_grad(self.final_norm, False)
        self._set_requires_grad(self.fc_out, False)
        if self.use_patient_embedding:
            self._set_requires_grad(self.patient_emb, False)

    def unfreeze_temporal(self):
        self._set_requires_grad(self.input_proj, True)
        self._set_requires_grad(self.mamba_layers, True)
        self._set_requires_grad(self.final_norm, True)
        self._set_requires_grad(self.fc_out, True)
        if self.use_patient_embedding:
            self._set_requires_grad(self.patient_emb, True)

    def save_autoencoder_weights(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "grid_size": self.decoder.out_size,
            },
            path,
        )

    def load_autoencoder_weights(self, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location)
        if "encoder" not in checkpoint or "decoder" not in checkpoint:
            raise ValueError(
                f"Autoencoder checkpoint {path} must contain 'encoder' and 'decoder' weights"
            )
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

    def forward_autoencoder(self, grids):
        batch_size, sequence_length = grids.shape[:2]
        flat_grids = grids.reshape(-1, *grids.shape[2:])
        reconstructed = self.decoder(self.encoder(flat_grids))
        return reconstructed.reshape(
            batch_size, sequence_length, *reconstructed.shape[1:]
        )

    def _encode_grids(self, grids):
        batch_size, sequence_length = grids.shape[:2]
        flat_grids = grids.reshape(-1, *grids.shape[2:])
        latents = self.encoder(flat_grids)
        return latents.reshape(batch_size, sequence_length, self.latent_dim)

    def _predict_latent(self, sequence_inputs):
        hidden = self.input_proj(sequence_inputs)
        for layer in self.mamba_layers:
            hidden = layer(hidden)
        return self.fc_out(self.final_norm(hidden[:, -1]))

    def forward(self, grids, times, patient_idx=None, n_future=0):
        if grids.ndim != 6:
            raise ValueError(
                "grids must have shape [batch, time, channels, depth, height, width]"
            )
        batch_size, sequence_length = grids.shape[:2]
        if sequence_length == 0:
            raise ValueError("At least one observed grid is required")
        if times.ndim != 2 or times.shape[0] != batch_size:
            raise ValueError("times must have shape [batch, time]")

        observed_latents = self._encode_grids(grids)
        available_steps = times.shape[1] - 1
        if n_future:
            available_steps = max(available_steps, sequence_length - 1 + n_future)
        if available_steps <= 0:
            return grids.new_empty((batch_size, 0, *grids.shape[2:]))
        if times.shape[1] < available_steps + 1:
            raise ValueError(
                "times must include a target timestamp for every requested prediction"
            )

        patient_embedding = None
        if self.use_patient_embedding:
            if patient_idx is None:
                raise ValueError("patient_idx is required when patient embeddings are enabled")
            patient_embedding = self.patient_emb(patient_idx)

        model_inputs = []
        predicted_latents = []
        predictions = []
        for step in range(available_steps):
            if step < sequence_length:
                current_latent = observed_latents[:, step]
            else:
                current_latent = predicted_latents[-1]
            time_encoding = self.time_encoder(times[:, step + 1])
            input_parts = [current_latent, time_encoding]
            if patient_embedding is not None:
                input_parts.append(patient_embedding)
            model_inputs.append(torch.cat(input_parts, dim=-1))

            predicted_latent = self._predict_latent(
                torch.stack(model_inputs, dim=1)
            )
            predicted_latents.append(predicted_latent)
            predictions.append(self.decoder(predicted_latent))

        return torch.stack(predictions, dim=1)
