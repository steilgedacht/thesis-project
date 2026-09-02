"""
Point-based Neural ODE model — drop-in replacement for the coordinate INR.

Design
------
Instead of feeding (x, y, z, t) jointly into one MLP, time is factored out
into a latent trajectory:

    z(0)  = g(patient_embedding)                     # initial latent state
    z(t)  = ODESolve(z(0), f_theta, 0 -> t)            # Neural ODE
    pred  = Decoder(x, y, z_spatial, z(t))             # occupancy logit

`f_theta` (LatentODEFunc) and the integrator are both plain PyTorch, no
external ODE library — a fixed-step RK4/Euler solver is used instead of
torchdiffeq, since every trajectory in a batch has a different target time
and this keeps things dependency-free and easy to debug.

Interface compatibility
------------------------
forward(coords, patient_idx) has the exact same signature/shape contract as
the original INR:
    coords:      [B, N, 4]  (x, y, z, t) — t is assumed constant across the
                  N points of one batch item (true for every dataset in
                  dataloader.py: LesionDataset always samples one
                  random_time_point per item and repeats it across all N
                  sampled points).
    patient_idx: [B]         patient/lesion embedding id, same convention.

    returns:     [B, N, 1]  occupancy logits.

Because of this, `dataloader.py`, `loss.py` and every plotting function in
`train_plotting.py` (the non-LSTM / INR path) work completely unchanged.

Total-variation loss note
--------------------------
`Loss_BCE_Dice_TV.compute_tv_losses` differentiates the prediction w.r.t.
the full `coords` tensor via autograd, including the time channel
`coords[..., 3]`. Since time is now shared across all N points of an item
(only ONE ODE integration per item, not one per point), this model reduces
that duplicated time channel with `mean(dim=1)` before it enters the graph
-- see `_shared_time` below. This keeps the value numerically identical
(all N entries are already equal) while making sure autograd distributes a
non-zero gradient back to every position in `coords[..., 3]`, not just
index 0. The resulting "time-TV" term now measures smoothness of the
*trajectory itself* over time, which is arguably a more natural fit for an
ODE than the original per-point interpretation.
"""

import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Standard NeRF-style positional encoding for the spatial input.
    Helps a coordinate-conditioned decoder represent sharp lesion
    boundaries instead of only smooth/low-frequency shapes. Set
    num_frequencies=0 to disable and fall back to raw (x, y, z).
    """
    def __init__(self, in_dim=3, num_frequencies=6, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_frequencies = num_frequencies
        if num_frequencies > 0:
            freq_bands = 2.0 ** torch.arange(num_frequencies)
            self.register_buffer("freq_bands", freq_bands)
        self.out_dim = in_dim * (2 * num_frequencies + (1 if include_input else 0))

    def forward(self, x):
        if self.num_frequencies == 0:
            return x
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq * math.pi))
            out.append(torch.cos(x * freq * math.pi))
        return torch.cat(out, dim=-1)


class LatentODEFunc(nn.Module):
    """f_theta(z, cond) -> dz/dt. `cond` (the patient embedding) is fed in
    at every ODE step so the dynamics themselves are patient-specific, not
    just the initial condition."""
    def __init__(self, latent_dim, cond_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        layers = [nn.Linear(latent_dim + cond_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, cond):
        return self.net(torch.cat([z, cond], dim=-1))


class FixedStepODESolver(nn.Module):
    """Dependency-free Euler / RK4 integrator.

    Every batch item integrates over its own physical time span
    [0, t_target_i]. To vectorize this as one Python loop over a *shared*
    number of steps, each item's step size is rescaled to
    dt_i = t_target_i / n_steps ("time-rescaling" trick). This is an
    approximation (equivalent to a change of integration variable
    tau = t / t_target_i, integrated uniformly in tau) but is exact for the
    fixed-step schemes here and is the standard way to batch latent-ODE
    training without an adaptive solver.
    """
    def __init__(self, ode_func, n_steps=16, method="rk4"):
        super().__init__()
        assert method in ("euler", "rk4"), "method must be 'euler' or 'rk4'"
        self.ode_func = ode_func
        self.n_steps = n_steps
        self.method = method

    def forward(self, z0, t_target, cond):
        # z0: [B, latent_dim], t_target: [B, 1], cond: [B, cond_dim]
        dt = t_target / self.n_steps
        z = z0
        for _ in range(self.n_steps):
            if self.method == "euler":
                z = z + dt * self.ode_func(z, cond)
            else:  # rk4
                k1 = self.ode_func(z, cond)
                k2 = self.ode_func(z + 0.5 * dt * k1, cond)
                k3 = self.ode_func(z + 0.5 * dt * k2, cond)
                k4 = self.ode_func(z + dt * k3, cond)
                z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z


class SpatialDecoder(nn.Module):
    """Maps (encoded x, y, z) + z(t) -> occupancy logit."""
    def __init__(self, spatial_in_dim, latent_dim, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(spatial_in_dim + latent_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xyz_encoded, z_t):
        return self.net(torch.cat([xyz_encoded, z_t], dim=-1))


class NeuralODE_INR(nn.Module):
    """
    Drop-in replacement for the coordinate-based INR model.

    Parameters
    ----------
    trajectories : accepted for interface-compatibility with the old
        `config.model(trajectories=trajectories, **config.model_params)`
        call in train.py; unused here.
    num_patients : int
        Size of the patient/lesion embedding table. train.py now passes
        this automatically (`len(train_dataset.patient_to_idx)` or similar)
        -- see the accompanying train.py change.
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
        spatial_hidden_dim=256,
        spatial_layers=4,
        num_fourier_frequencies=6,
    ):
        super().__init__()

        self.patient_embedding = nn.Embedding(num_patients, patient_embed_dim)

        # z(0), derived from the patient embedding.
        self.initial_state_net = nn.Sequential(
            nn.Linear(patient_embed_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

        ode_func = LatentODEFunc(
            latent_dim=latent_dim,
            cond_dim=patient_embed_dim,
            hidden_dim=ode_hidden_dim,
            n_layers=ode_layers,
        )
        self.ode_solver = FixedStepODESolver(ode_func, n_steps=ode_steps, method=ode_method)

        self.spatial_encoding = FourierFeatures(in_dim=3, num_frequencies=num_fourier_frequencies)
        self.decoder = SpatialDecoder(
            spatial_in_dim=self.spatial_encoding.out_dim,
            latent_dim=latent_dim,
            hidden_dim=spatial_hidden_dim,
            n_layers=spatial_layers,
        )

    @staticmethod
    def _shared_time(coords):
        # coords[..., 3]: [B, N], identical across N for every dataset in
        # dataloader.py. mean() keeps the value unchanged but lets autograd
        # distribute gradient to every position (see module docstring).
        return coords[..., 3].mean(dim=1, keepdim=True)  # [B, 1]

    def forward(self, coords, patient_idx):
        xyz = coords[..., :3]                 # [B, N, 3]
        t = self._shared_time(coords)          # [B, 1]

        patient_emb = self.patient_embedding(patient_idx)   # [B, E]
        z0 = self.initial_state_net(patient_emb)             # [B, latent_dim]
        z_t = self.ode_solver(z0, t, patient_emb)             # [B, latent_dim]

        N = xyz.shape[1]
        z_t_expanded = z_t.unsqueeze(1).expand(-1, N, -1)     # [B, N, latent_dim]

        xyz_encoded = self.spatial_encoding(xyz)               # [B, N, spatial_in_dim]
        logits = self.decoder(xyz_encoded, z_t_expanded)        # [B, N, 1]
        return logits