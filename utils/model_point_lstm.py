"""Point-LSTM: observation-conditioned lesion forecasting with an implicit decoder.

    scan_i (sampled points) --PointSetEncoder--> e_i
    LSTM over [e_i, tau(dt_i), tau(t_i), log volume_i]          -> h_k
    head([h_k, tau(t* - t_k), tau(t*) (, lesion embedding)])     -> z*
    SpatialDecoder(Fourier(xyz) (, sdf), z*)                     -> occupancy logit

The decoder, Fourier features and coordinate frame are the same as in the
Neural ODE, so the two models differ only in how z(t) is obtained: the ODE
integrates a per-lesion embedding, the LSTM reads the observed scans. New
lesions therefore need no optimization, only their observed scans.

Options
-------
use_sdf_prior:
    The decoder additionally receives the signed distance (mm) from each query
    point to the boundary of the last observed lesion, and the logit gets a
    learnable prior term ``-gain * sdf``. At initialization the model therefore
    reproduces the last observation; the network learns how the boundary moves.
use_lesion_embedding:
    Optional per-lesion vector for the forecast head, zero-initialized and
    randomly dropped during training, so the model always works without it
    (unseen lesions keep a zero vector) and can be fine-tuned per lesion.

Interface
---------
forward(coords, patient_idx, history, sdf=None)
    coords:  [B, N, 4] query xyz + target time (time shared across N)
    history: dict(points [B, L, P, 4], times [B, L], log_volume [B, L], mask [B, L])
    sdf:     [B, N, 1] signed distance to the last history lesion (mm), required
             if use_sdf_prior
For dense inference: context = encode_history(history); z = latent(context,
t, patient_idx); then decode(coords_chunk, z, sdf_chunk) per chunk.
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .model_neural_ode import FourierFeatures, SpatialDecoder


class TimeFeatures(nn.Module):
    """[log1p(t / 1 year), t / max_t, sin/cos(pi 2^k t / max_t)]."""

    def __init__(self, max_t=3650.0, num_frequencies=4):
        super().__init__()
        self.max_t = max_t
        self.register_buffer("freqs", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32))
        self.out_dim = 2 + 2 * num_frequencies

    def forward(self, t):
        t = t.unsqueeze(-1)
        x = t / self.max_t
        projections = x * self.freqs * math.pi
        return torch.cat(
            [torch.log1p(t.clamp_min(0.0) / 365.0), x, torch.sin(projections), torch.cos(projections)],
            dim=-1,
        )


class PointSetEncoder(nn.Module):
    """DeepSets/PointNet encoder: shared point MLP, max+mean pooling."""

    def __init__(self, num_fourier_frequencies=8, hidden_dim=256, n_layers=3, out_dim=256):
        super().__init__()
        self.spatial_encoding = FourierFeatures(in_dim=3, num_frequencies=num_fourier_frequencies)
        layers, in_dim = [], self.spatial_encoding.out_dim + 1
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.point_mlp = nn.Sequential(*layers)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, points):
        xyz, label = points[..., :3], points[..., 3:4]
        features = self.point_mlp(torch.cat([self.spatial_encoding(xyz), 2.0 * label - 1.0], dim=-1))
        pooled = torch.cat([features.max(dim=-2).values, features.mean(dim=-2)], dim=-1)
        return self.out(pooled)


class PointLSTM(nn.Module):
    def __init__(
        self,
        trajectories=None,              # interface compatibility, unused
        num_patients=1,
        num_fourier_frequencies=8,
        encoder_hidden_dim=256,
        encoder_layers=3,
        scan_code_dim=256,
        lstm_hidden_dim=256,
        lstm_layers=2,
        latent_dim=256,
        head_hidden_dim=256,
        spatial_hidden_dim=256,
        spatial_layers=8,
        time_frequencies=4,
        max_t=3650.0,
        history_points=1024,
        max_history_len=5,
        use_sdf_prior=False,
        sdf_clip_mm=20.0,
        use_lesion_embedding=False,
        lesion_embed_dim=32,
        lesion_embedding_dropout=0.5,
    ):
        super().__init__()
        # Read by the data side (training script and validation arena).
        self.history_points = history_points
        self.max_history_len = max_history_len
        self.use_sdf_prior = use_sdf_prior
        self.sdf_clip_mm = float(sdf_clip_mm)
        self.use_lesion_embedding = use_lesion_embedding
        self.lesion_embedding_dropout = lesion_embedding_dropout

        self.encoder = PointSetEncoder(
            num_fourier_frequencies, encoder_hidden_dim, encoder_layers, scan_code_dim
        )
        self.time_features = TimeFeatures(max_t, time_frequencies)
        time_dim = self.time_features.out_dim

        self.lstm = nn.LSTM(
            scan_code_dim + 2 * time_dim + 1, lstm_hidden_dim,
            num_layers=lstm_layers, batch_first=True,
        )

        head_in = lstm_hidden_dim + 2 * time_dim
        if use_lesion_embedding:
            self.lesion_embedding = nn.Embedding(num_patients, lesion_embed_dim)
            nn.init.zeros_(self.lesion_embedding.weight)
            head_in += lesion_embed_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden_dim), nn.SiLU(),
            nn.Linear(head_hidden_dim, head_hidden_dim), nn.SiLU(),
            nn.Linear(head_hidden_dim, latent_dim),
        )

        self.spatial_encoding = FourierFeatures(in_dim=3, num_frequencies=num_fourier_frequencies)
        self.decoder = SpatialDecoder(
            spatial_in_dim=self.spatial_encoding.out_dim + (1 if use_sdf_prior else 0),
            latent_dim=latent_dim,
            hidden_dim=spatial_hidden_dim,
            n_layers=spatial_layers,
        )
        if use_sdf_prior:
            self.sdf_prior_gain = nn.Parameter(torch.tensor(1.0))

    # -- history -> context --------------------------------------------------
    def encode_history(self, history):
        points, times, mask = history["points"], history["times"], history["mask"]
        lengths = mask.sum(dim=1)
        if bool((lengths < 1).any()):
            raise ValueError("Every item needs at least one history scan")
        codes = self.encoder(points)                                   # [B, L, C]
        previous = torch.cat([times[:, :1], times[:, :-1]], dim=1)
        dt = (times - previous).clamp_min(0.0)
        steps = torch.cat(
            [codes, self.time_features(dt), self.time_features(times),
             (history["log_volume"] / 10.0).unsqueeze(-1)],
            dim=-1,
        )
        packed = pack_padded_sequence(steps, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        last_time = times.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        return {"state": h_n[-1], "last_time": last_time}

    def latent(self, context, target_time, patient_idx=None):
        dt = (target_time - context["last_time"]).clamp_min(0.0)
        features = [context["state"], self.time_features(dt), self.time_features(target_time)]
        if self.use_lesion_embedding:
            embedding = self.lesion_embedding(patient_idx)
            if self.training and self.lesion_embedding_dropout > 0:
                keep = (torch.rand(embedding.shape[0], 1, device=embedding.device)
                        >= self.lesion_embedding_dropout).to(embedding.dtype)
                embedding = embedding * keep
            features.append(embedding)
        return self.head(torch.cat(features, dim=-1))

    def decode(self, coords, z, sdf=None):
        encoded = self.spatial_encoding(coords[..., :3])
        if self.use_sdf_prior:
            if sdf is None:
                raise ValueError("use_sdf_prior=True requires the sdf input")
            sdf = sdf.clamp(-self.sdf_clip_mm, self.sdf_clip_mm)
            encoded = torch.cat([encoded, sdf / self.sdf_clip_mm], dim=-1)
        z = z.unsqueeze(1).expand(-1, coords.shape[1], -1)
        logits = self.decoder(encoded, z)
        if self.use_sdf_prior:
            logits = logits - self.sdf_prior_gain * sdf
        return logits

    def forward(self, coords, patient_idx, history, sdf=None):
        context = self.encode_history(history)
        target_time = coords[..., 3].mean(dim=1)
        z = self.latent(context, target_time, patient_idx)
        return self.decode(coords, z, sdf)
