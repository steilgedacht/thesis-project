"""Lesion time-evolution GIF for the Point-LSTM.

Same six-panel animation as ``utils/train_plotting.plot_lesion_time_evolution``
(heatmap, label slice, 3D labels, overlap, size timeline, top-down map), and
the same query slice: the label grid's full x/y extent at the lesion's z,
mapped to model coordinates exactly like the INR / Neural ODE plot.

The Point-LSTM is a forecaster, so each frame time ``t`` is conditioned on
the observed scans strictly before ``t`` (at most ``max_history_len``, the
most recent ones; before the first scan the first scan is used). Held-out
validation dates (extrapolation + interpolation) are never used as history,
so frames after them are genuine forecasts. Frames sharing the same history
reuse one history encoding.

The heatmap shows sigmoid probabilities in [0, 1] (not peak-normalized), so
an empty prediction stays dark.
"""

import zlib

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from .dataloader_point_lstm import (
    all_dates,
    batch_to_device,
    collate_history,
    held_out_dates,
    history_arrays,
    lesion_key,
)
from .train_plotting import (
    LesionTimeSeries,
    _build_figure,
    _build_query_grid,
    _build_topdown_height_map,
    _compute_lesion_z_voxel,
    _compute_reference_geometry,
    _find_label_at_time,
    _heatmap_pixel_spacing,
    _load_full_matrix_with_affine,
    _save_gif,
    _update_frame,
)


def _observed_entries(trj, cache, validation_samples):
    """ScanEntries usable as history (held-out validation dates excluded), by time."""
    held = held_out_dates(trj, validation_samples)
    entries = [cache.get(trj, d) for d in all_dates(trj) if d not in held]
    return sorted(entries, key=lambda entry: entry.time)


@torch.no_grad()
def _predict_heatmaps(model, trj, cache, validation_samples, embedding_id, time_points,
                      xyz, side_length, ring_mm, device, chunk_size=262144):
    observed = _observed_entries(trj, cache, validation_samples)
    if not observed:
        raise ValueError(f"{lesion_key(trj)}: no observed scans to condition on.")
    patient_idx = torch.as_tensor([int(embedding_id)], device=device)
    xyz_tensor = torch.from_numpy(xyz.astype(np.float32)).unsqueeze(0).to(device)

    heatmaps, contexts = [], {}
    for t in time_points:
        history = [e for e in observed if e.time < t] or observed[:1]
        history = history[-model.max_history_len:]
        key = tuple(e.time for e in history)
        if key not in contexts:  # deterministic history sampling, like validation
            rng = np.random.default_rng(zlib.crc32(f"{lesion_key(trj)}|{key}".encode()))
            batch = batch_to_device(
                collate_history([history_arrays(history, model.history_points, rng, ring_mm)]),
                device,
            )
            contexts[key] = model.encode_history(batch)
        target_time = torch.tensor([float(t)], dtype=torch.float32, device=device)
        z = model.latent(contexts[key], target_time, patient_idx)

        probabilities = []
        for start in range(0, xyz.shape[0], chunk_size):
            chunk = xyz_tensor[:, start:start + chunk_size]
            sdf = None
            if model.use_sdf_prior:
                sdf = torch.from_numpy(history[-1].sdf_at_model(xyz[start:start + chunk_size]))
                sdf = sdf.view(1, -1, 1).to(device)
            coords = torch.cat([chunk, torch.full_like(chunk[..., :1], float(t))], dim=-1)
            probabilities.append(torch.sigmoid(model.decode(coords, z, sdf)).reshape(-1).cpu())
        heatmaps.append(torch.cat(probabilities).numpy().reshape(side_length, side_length))
    return np.asarray(heatmaps)


def plot_point_lstm_time_evolution(model, epoch, trj, cache, validation_samples, embedding_id,
                                   config, final_side_length=False, ring_mm=3.0):
    """Render the time-evolution GIF for one lesion and log it to MLflow."""
    side_length = (
        getattr(config, "full_size_side_length", 500) if final_side_length
        else getattr(config, "time_evolution_side_length", 50)
    )
    was_training = model.training
    model.eval()

    full_matrix_labels, affine = _load_full_matrix_with_affine(trj)
    label_shape = np.array(full_matrix_labels[0][0].shape)
    geometry = _compute_reference_geometry(label_shape, affine)
    lesion_z_voxel = _compute_lesion_z_voxel(full_matrix_labels)
    time_points = np.linspace(
        0, full_matrix_labels[-1][1] + 180, getattr(config, "time_evolution_steps", 100)
    )
    x, y, z = _build_query_grid(label_shape, affine, geometry, lesion_z_voxel, side_length)
    heatmaps = _predict_heatmaps(
        model, trj, cache, validation_samples, embedding_id, time_points,
        np.stack([x, y, z], axis=1), side_length, ring_mm, config.device,
    )

    label_slices, volumes_3d, lesion_sizes = [], [], []
    for t in time_points:
        volume, label_slice = _find_label_at_time(full_matrix_labels, t, lesion_z_voxel)
        volumes_3d.append(volume)
        label_slices.append(label_slice)
        lesion_sizes.append(float(volume.sum()))
    volumes_3d = np.array(volumes_3d)
    lesion_sizes_cm3 = np.array(lesion_sizes) * np.prod(geometry.voxel_spacing) / 1000.0
    change_points = np.zeros(len(time_points), dtype=bool)
    change_points[1:] = lesion_sizes_cm3[1:] != lesion_sizes_cm3[:-1]
    spacing_i, spacing_j = _heatmap_pixel_spacing(geometry, label_shape, side_length)

    data = LesionTimeSeries(
        time_points=time_points,
        heatmaps=heatmaps,
        heatmap_aspect=spacing_j / spacing_i,
        heatmap_pixel_spacing_mm=(spacing_i, spacing_j),
        label_slices=np.array(label_slices),
        volumes_3d=volumes_3d,
        topdown_maps=[_build_topdown_height_map(volume) for volume in volumes_3d],
        lesion_sizes_cm3=lesion_sizes_cm3,
        change_points=change_points,
        lesion_z_voxel=lesion_z_voxel,
        geometry=geometry,
    )

    patient_idx = int(embedding_id)
    artists = _build_figure(data, patient_idx)
    output_path = (
        f"/tmp/{epoch:04d}_epoch_point_lstm_heatmap_patient_"
        f"{trj.patient_id}_{patient_idx}_lesion_{trj.label_id}.gif"
    )
    _save_gif(artists.fig, lambda i: _update_frame(i, data, artists, patient_idx),
              len(heatmaps), output_path)
    mlflow.log_artifact(output_path)
    plt.close(artists.fig)
    if was_training:
        model.train()
