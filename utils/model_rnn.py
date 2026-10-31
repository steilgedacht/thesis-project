from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .model_lstm import Decoder3D, Encoder3D, TimeEncoder


class LesionLatentRNN(nn.Module):
    """Encode lesion masks and forecast their latent sequence with a vanilla RNN."""

    def __init__(
        self,
        trajectories,
        latent_dim=256,
        hidden_dim=256,
        n_layers=2,
        dropout=0.0,
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
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

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
        self._set_requires_grad(self.rnn, False)
        self._set_requires_grad(self.final_norm, False)
        self._set_requires_grad(self.fc_out, False)
        if self.use_patient_embedding:
            self._set_requires_grad(self.patient_emb, False)

    def unfreeze_temporal(self):
        self._set_requires_grad(self.input_proj, True)
        self._set_requires_grad(self.rnn, True)
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
        recurrent_inputs = self.input_proj(sequence_inputs)
        recurrent_outputs, _ = self.rnn(recurrent_inputs)
        return self.fc_out(self.final_norm(recurrent_outputs[:, -1]))

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
                raise ValueError(
                    "patient_idx is required when patient embeddings are enabled"
                )
            patient_embedding = self.patient_emb(patient_idx)

        model_inputs = []
        predicted_latents = []
        predictions = []
        for step in range(available_steps):
            current_latent = (
                observed_latents[:, step]
                if step < sequence_length
                else predicted_latents[-1]
            )
            input_parts = [
                current_latent,
                self.time_encoder(times[:, step + 1]),
            ]
            if patient_embedding is not None:
                input_parts.append(patient_embedding)
            model_inputs.append(torch.cat(input_parts, dim=-1))

            predicted_latent = self._predict_latent(torch.stack(model_inputs, dim=1))
            predicted_latents.append(predicted_latent)
            predictions.append(self.decoder(predicted_latent))

        return torch.stack(predictions, dim=1)
