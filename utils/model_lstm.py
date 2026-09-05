import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeEncoder(nn.Module):
    """Identical sinusoidal encoding to the INR baseline, for a fair comparison."""
    def __init__(self, max_t, num_frequencies=6):
        super().__init__()
        self.max_t = max_t
        self.register_buffer(
            "frequencies", 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        )

    def forward(self, t):
        # t: [B] or [B, 1] -> returns [B, 2 * num_frequencies]
        t_norm = t.view(-1, 1) / self.max_t
        angles = t_norm * self.frequencies * np.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FiLMConvLSTMCell(nn.Module):
    """
    A ConvLSTM cell whose gate pre-activations are FiLM-modulated by a
    conditioning vector (patient latent + encoded target-time), mirroring the
    gamma/beta modulation used in SirenLayer of the INR model. This keeps the
    conditioning mechanism analogous between the two models, which makes the
    comparison more meaningful (same "how do we inject patient/time info"
    strategy, different backbone).
    """
    def __init__(self, in_channels, hidden_channels, cond_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # Standard ConvLSTM gates: input, forget, output, cell candidate -> 4 * hidden
        self.conv = nn.Conv3d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        # FiLM conditioning applied to the 4*hidden pre-activation maps
        self.cond_proj = nn.Linear(cond_dim, 2 * 4 * hidden_channels)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        with torch.no_grad():
            # start close to an identity modulation (gamma=1, beta=0)
            self.cond_proj.bias[: 4 * hidden_channels] = 1.0

    def forward(self, x, cond, h_prev, c_prev):
        # x, h_prev, c_prev: [B, C, D, H, W]
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)  # [B, 4*hidden, D, H, W]

        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)  # each [B, 4*hidden]
        gamma = gamma.view(*gamma.shape, 1, 1, 1)
        beta = beta.view(*beta.shape, 1, 1, 1)
        gates = gamma * gates + beta

        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size, device):
        d, h, w = spatial_size
        shape = (batch_size, self.hidden_channels, d, h, w)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)


class LesionLSTM(nn.Module):
    """
    Grid-based recurrent baseline for the same task as LesionINR: predict
    per-voxel lesion occupancy over time for a patient.

    Unlike the INR (which queries a continuous field at arbitrary (x,y,z,t)),
    this model operates on a fixed-resolution occupancy grid per visit.
    Predict-the-whole-grid-then-roll-forward is the right pattern here: a
    ConvLSTM step consumes the full spatial state at visit t and produces the
    full spatial state at visit t+1 in one shot, which is both spatially
    coherent and directly interpretable (threshold it, compute Dice, etc.).

    Training uses teacher forcing (real grid_t -> predict grid_{t+1} for every
    consecutive pair). Forecasting beyond the observed visits is autoregressive:
    the model's own prediction is fed back in as the next input.

    Default grid_size=(64,64,64) is a compromise for memory; drop resolution
    further or swap Conv3d->Conv2d (2D per-slice, e.g. 100x100) if needed.
    """
    def __init__(self,
                 trajectories,
                 grid_size=(64, 64, 64),
                 latent_dim=128,
                 hidden_channels=32,
                 n_layers=2,
                 time_freqs=6,
                 max_t=3650.0,
                 kernel_size=3):
        super().__init__()
        num_patients = len(trajectories) * 100  # same collision-avoidance trick as the INR
        self.grid_size = grid_size
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.latent_vectors = nn.Embedding(num_patients, latent_dim)
        torch.nn.init.normal_(self.latent_vectors.weight, std=1.0 / np.sqrt(latent_dim))

        self.time_encoder = TimeEncoder(max_t=max_t, num_frequencies=time_freqs)
        cond_dim = latent_dim + 2 * time_freqs

        # Lift the 1-channel occupancy grid into feature space
        self.in_proj = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)

        self.cells = nn.ModuleList([
            FiLMConvLSTMCell(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                cond_dim=cond_dim,
                kernel_size=kernel_size,
            )
            for _ in range(n_layers)
        ])

        # Project hidden state back to a 1-channel occupancy logit map
        self.out_proj = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def _step(self, grid, cond, hidden_states):
        """One recurrent step: grid_t, cond -> (updated hidden states, predicted grid_{t+1} logits)"""
        x = self.in_proj(grid)
        new_hidden = []
        for cell, (h, c) in zip(self.cells, hidden_states):
            h, c = cell(x, cond, h, c)
            x = h  # feed to next layer
            new_hidden.append((h, c))
        pred = self.out_proj(x)
        return new_hidden, pred

    def init_hidden(self, batch_size, device):
        return [cell.init_hidden(batch_size, self.grid_size, device) for cell in self.cells]

    def forward(self, grids, times, patient_idx, teacher_forcing=True, n_future=0):
        """
        grids:       [B, T, 1, D, H, W]  observed occupancy grids at T visits
        times:       [B, T (+n_future)]  raw timestamps (same units as max_t) per visit.
                     If n_future > 0, must include n_future extra future
                     timestamps appended after the T observed ones.
        patient_idx: [B]                 patient id, same convention as the INR
        teacher_forcing: if True, feed the *true* grid at each step within the
                     observed sequence (training). If False, feed the model's
                     own previous prediction instead (validation/inference).
        n_future:    additional autoregressive steps to roll out *beyond* the
                     observed sequence, e.g. to forecast unseen future visits.

        Returns:
          preds: [B, (T-1) + n_future, 1, D, H, W] logits, one per predicted
                 step (predicting visit t+1 from visit t, for t=1..T-1, then
                 n_future more autoregressive steps). Apply sigmoid for
                 occupancy probabilities.
        """
        B, T = grids.shape[0], grids.shape[1]
        device = grids.device

        z_patient = self.latent_vectors(patient_idx)  # [B, latent_dim]
        hidden = self.init_hidden(B, device)
        preds = []

        current_input = grids[:, 0]  # first observed grid is always real
        total_steps = (T - 1) + n_future

        for step in range(total_steps):
            # condition on the time of the visit we're predicting *toward*
            target_time = times[:, step + 1]
            t_enc = self.time_encoder(target_time)
            cond = torch.cat([z_patient, t_enc], dim=-1)

            hidden, pred_logits = self._step(current_input, cond, hidden)
            preds.append(pred_logits)

            if step + 1 < T:
                # still within the observed sequence
                current_input = grids[:, step + 1] if teacher_forcing else torch.sigmoid(pred_logits)
            else:
                # forecasting beyond observed data: must use our own prediction
                current_input = torch.sigmoid(pred_logits)

        return torch.stack(preds, dim=1)

class ResBlock3D(nn.Module):
    """Intra-block 3D Residual Block (Identity + Conv -> LeakyReLU -> Conv).
    
    Operates strictly inside the block (no cross-network UNet skip connections).
    Preserves fine spatial features and stabilizes gradient flow.
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class UpResBlock3D(nn.Module):
    """Upsampling Block: Nearest-Neighbor Interpolation -> Conv (Channel Change) -> ResBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_match = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.res_block = ResBlock3D(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_match(x)
        x = self.res_block(x)
        return x


class Encoder3D(nn.Module):
    """3D conv encoder mapping a 1-channel occupancy grid to a latent vector."""
    def __init__(self, in_channels=1, latent_dim=256, base_channels=32, grid_size=(64, 64, 64)):
        super().__init__()
        self.grid_size = grid_size
        self.bottleneck_size = tuple(max(1, s // 16) for s in grid_size)  # 4 downsample steps = 16x

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),          # /2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),      # /4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # /8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1),  # /16
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(self.bottleneck_size)

        flat_dim = base_channels * 4 * self.bottleneck_size[0] * self.bottleneck_size[1] * self.bottleneck_size[2]
        self.fc = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = h.view(h.shape[0], -1)
        return self.fc(h)


class Decoder3D(nn.Module):
    """3D Decoder using intra-block residual connections and nearest upsampling."""
    def __init__(self, latent_dim=256, out_channels=1, base_channels=32, out_size=(64, 64, 64)):
        super().__init__()
        self.out_size = out_size
        self.bottleneck_size = tuple(max(1, s // 16) for s in out_size)  # Matches Encoder's /16

        flat_dim = base_channels * 4 * self.bottleneck_size[0] * self.bottleneck_size[1] * self.bottleneck_size[2]
        self.fc = nn.Linear(latent_dim, flat_dim)

        self.unflatten = nn.Unflatten(1, (base_channels * 4, *self.bottleneck_size))
        
        # Upsampling with intra-block Residuals
        self.layer1 = UpResBlock3D(base_channels * 4, base_channels * 4)  # x2
        self.layer2 = UpResBlock3D(base_channels * 4, base_channels * 2)  # x4
        self.layer3 = UpResBlock3D(base_channels * 2, base_channels)      # x8
        self.layer4 = UpResBlock3D(base_channels, base_channels)          # x16
        
        # Final projection to target channels (output logits or probabilities)
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.fc(z)
        h = self.unflatten(h)
        
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        
        out = self.final_conv(h)
        
        if out.shape[-3:] != self.out_size:
            out = F.interpolate(out, size=self.out_size, mode='trilinear', align_corners=False)
            
        return out

class LesionLatentLSTM(nn.Module):
    """
    Latent-space recurrent model:
      encoder: grid -> z
      latent-LSTM (FiLM-style conditioning on time + optional patient embedding)
      decoder: z -> grid

    This keeps a similar conditioning scheme (concatenate time encoding to z)
    and uses a small RNN over latent vectors instead of ConvLSTM over
    full grids, which reduces memory and lets the autoencoder learn compact
    visit representations.

    Supports staged training:
      - `forward_autoencoder(grids)` runs encoder+decoder independently per
        frame (no RNN at all), for an autoencoder-only pretraining phase.
      - `freeze_autoencoder()` / `unfreeze_autoencoder()` toggle
        requires_grad on the encoder+decoder.
      - `freeze_temporal()` / `unfreeze_temporal()` toggle requires_grad on
        the RNN, fc_out, and (if used) the patient embedding.
    """
    def __init__(self, trajectories, latent_dim=128, hidden_dim=256, n_layers=2, time_freqs=6,
                 max_t=3650.0, encoder_params=None, decoder_params=None,
                 use_patient_embedding=False, grid_size=(64, 64, 64)):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_patient_embedding = use_patient_embedding

        num_patients = len(trajectories) * 100
        if use_patient_embedding:
            self.patient_emb = nn.Embedding(num_patients, latent_dim)
            torch.nn.init.normal_(self.patient_emb.weight, std=1.0 / np.sqrt(latent_dim))

        encoder_params = dict(encoder_params or {})
        encoder_params.setdefault("grid_size", grid_size)
        decoder_params = dict(decoder_params or {})
        decoder_params.setdefault("out_size", grid_size)

        self.encoder = Encoder3D(latent_dim=latent_dim, **encoder_params)
        self.decoder = Decoder3D(latent_dim=latent_dim, **decoder_params)

        self.time_encoder = TimeEncoder(max_t=max_t, num_frequencies=time_freqs)
        lstm_input_dim = latent_dim + 2 * time_freqs + (latent_dim if use_patient_embedding else 0)
        # NOTE: was nn.GRU -- switched to a proper LSTM as requested.
        self.rnn = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    # ---------------------------------------------------------------
    # Staged-training helpers
    # ---------------------------------------------------------------
    @staticmethod
    def _set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def freeze_autoencoder(self):
        self._set_requires_grad(self.encoder, False)
        self._set_requires_grad(self.decoder, False)

    def unfreeze_autoencoder(self):
        self._set_requires_grad(self.encoder, True)
        self._set_requires_grad(self.decoder, True)

    def freeze_temporal(self):
        self._set_requires_grad(self.rnn, False)
        self._set_requires_grad(self.fc_out, False)
        if self.use_patient_embedding:
            self._set_requires_grad(self.patient_emb, False)

    def unfreeze_temporal(self):
        self._set_requires_grad(self.rnn, True)
        self._set_requires_grad(self.fc_out, True)
        if self.use_patient_embedding:
            self._set_requires_grad(self.patient_emb, True)

    def forward_autoencoder(self, grids):
        """
        Encode and decode every observed frame independently -- no RNN
        involved at all. Used for the autoencoder-only pretraining phase, so
        the encoder/decoder learn a faithful per-frame reconstruction before
        the (initially random, noisy) temporal model starts pushing
        gradients through them.

        grids: [B, T, 1, D, H, W]
        returns: reconstructions [B, T, 1, D, H, W] logits
        """
        B, T = grids.shape[0], grids.shape[1]
        recons = []
        for t in range(T):
            z = self.encoder(grids[:, t])
            recon = self.decoder(z)
            recons.append(recon)
        return torch.stack(recons, dim=1)

    # ---------------------------------------------------------------

    def forward(self, grids, times, patient_idx=None, n_future=0):
        # grids: [B, T, 1, D, H, W]
        B, T = grids.shape[0], grids.shape[1]
        device = grids.device

        # encode each observed grid into z
        z_obs = []
        for t in range(T):
            grid_t = grids[:, t]
            z_t = self.encoder(grid_t)  # [B, latent_dim]
            z_obs.append(z_t)
        z_obs = torch.stack(z_obs, dim=1)  # [B, T, latent_dim]

        total_steps = (T - 1) + n_future

        # build inputs to RNN: for each prediction step we condition on the target time
        inputs = []
        for step in range(total_steps):
            target_time = times[:, step + 1]
            t_enc = self.time_encoder(target_time)  # [B, 12]
            z_curr = z_obs[:, step]
            if self.use_patient_embedding and patient_idx is not None:
                p_emb = self.patient_emb(patient_idx)
                rnn_in = torch.cat([z_curr, t_enc, p_emb], dim=-1)
            else:
                rnn_in = torch.cat([z_curr, t_enc], dim=-1)
            inputs.append(rnn_in)

        rnn_in = torch.stack(inputs, dim=1)  # [B, total_steps, input_dim]
        out, _ = self.rnn(rnn_in)            # [B, total_steps, hidden_dim]  (LSTM returns (h_n, c_n) as the 2nd item)
        z_pred = self.fc_out(out)            # [B, total_steps, latent_dim]

        # decode predicted latents to predicted grids
        preds = []
        for s in range(z_pred.shape[1]):
            g = self.decoder(z_pred[:, s])
            preds.append(g)
        preds = torch.stack(preds, dim=1)  # [B, total_steps, 1, D, H, W]
        return preds