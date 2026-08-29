"""
Meta-learning-specific overrides for train_plotting.py.

`LesionINR` (model_inr_meta.py) only takes coordinates as input --
`forward(self, x)`. Patient identity is now captured by adapting the shared
meta-weights (theta_meta) via a MAML-style inner loop, not by a patient_idx
input. The plotting helpers in train_plotting.py predate this and:

  1. Call `model(coords, patient_idx)` -- crashes, since LesionINR.forward
     has no second parameter.
  2. Plot the raw, unadapted meta-weights -- even once (1) is fixed, this
     produces the shared prior, not a patient-specific prediction (see the
     NOTE in train_meta.py's train_inr).

Only `_predict_heatmap`, `_assemble_time_series`, and
`plot_lesion_time_evolution` need to change to fix both. Everything else
(figure building, frame updates, gif export, dataclasses, geometry helpers)
is reused unmodified from train_plotting.py.

Usage: in train_meta.py, swap
    from utils.train_plotting import *
for
    from utils.train_plotting_meta import *
No other changes needed -- train_meta.py only calls `plot_lesion_time_evolution`
directly.
"""
import numpy as np
import torch
import torch.optim as optim
import higher
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .train_plotting import (
    _load_full_matrix_with_affine,
    _compute_reference_geometry,
    _compute_lesion_z_voxel,
    _voxel_to_model_space,
    _build_query_grid,
    _find_label_at_time,
    _heatmap_pixel_spacing,
    _build_figure,
    _save_gif,
    _update_frame,
    LesionTimeSeries,
)


# ---------------------------------------------------------------------------
# New: build a per-patient support set for test-time adaptation
# ---------------------------------------------------------------------------

def _build_support_set_from_labels(full_matrix_labels, affine, geometry, config, max_background_per_scan=None):
    """
    Build a (coords, labels) support set for test-time inner-loop adaptation,
    from the patient's own real label volumes (the same ones already loaded
    for the ground-truth panels).

    ASSUMPTION: coordinate normalization mirrors LesionDataset.prepare_data
    via `_voxel_to_model_space` (same function the query-grid heatmap uses),
    and the lesion/background split follows `config.background_samples`.
    This does NOT reproduce `config.dialation_iterations` -- the dataset's
    dilation step isn't available here since we only have raw label volumes,
    not the dataset object. If adapted predictions look too tight around the
    lesion boundary, port the dilation step over from LesionDataset.
    """
    max_background_per_scan = max_background_per_scan or config.background_samples

    all_coords, all_labels = [], []
    for volume, day in full_matrix_labels:
        lesion_voxels = np.stack(np.where(volume == 1), axis=1)
        background_voxels_all = np.stack(np.where(volume == 0), axis=1)

        if len(background_voxels_all) > 0:
            n_bg = min(max_background_per_scan, len(background_voxels_all))
            bg_idx = np.random.choice(len(background_voxels_all), size=n_bg, replace=False)
            background_voxels = background_voxels_all[bg_idx]
        else:
            background_voxels = background_voxels_all

        if len(lesion_voxels) == 0 and len(background_voxels) == 0:
            continue

        voxel_ijk = np.concatenate([lesion_voxels, background_voxels], axis=0).astype(float)
        xyz = _voxel_to_model_space(voxel_ijk, affine, geometry.volume_center_phys)
        t_col = np.full((xyz.shape[0], 1), day, dtype=float)
        coords = np.concatenate([xyz, t_col], axis=1)

        labels = np.concatenate([
            np.ones(len(lesion_voxels), dtype=float),
            np.zeros(len(background_voxels), dtype=float),
        ])

        all_coords.append(coords)
        all_labels.append(labels)

    coords = np.concatenate(all_coords, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return coords, labels


# ---------------------------------------------------------------------------
# Override: model no longer takes patient_idx -- adapts via weights instead
# ---------------------------------------------------------------------------

def _predict_heatmap(model, t, x, y, z, side_length, batch_size=150_000):
    """Query the INR for one timepoint, in chunks to bound memory use.

    Overridden from train_plotting.py: drops the `patient_idx_tensor`
    argument to `model(...)`. LesionINR.forward(self, x) takes only
    coordinates -- the original call `model(chunk, patient_idx_tensor)`
    raises TypeError since there's no second parameter to receive it.
    `model` here is expected to already be the patient-adapted fast-weights
    model (see plot_lesion_time_evolution below), not the raw meta-model.
    """
    time_col = np.full_like(x, t)
    coords = np.stack([x, y, z, time_col], axis=1)
    coords_tensor = torch.from_numpy(coords).float().to(next(model.parameters()).device).unsqueeze(0)

    chunks = []
    with torch.no_grad():
        for start in range(0, coords_tensor.shape[1], batch_size):
            chunk = coords_tensor[:, start:start + batch_size]
            chunks.append(model(chunk).cpu().numpy())
    preds = np.concatenate(chunks, axis=1)
    return preds.reshape(side_length, side_length)


def _assemble_time_series(model, config, full_matrix_labels, label_shape, affine, geometry, side_length) -> LesionTimeSeries:
    """Overridden from train_plotting.py: no longer threads a
    `patient_idx_tensor` through to `_predict_heatmap` (see above). `model`
    is expected to already be patient-adapted. Everything past the
    prediction loop is identical to the original."""
    lesion_z_voxel = _compute_lesion_z_voxel(full_matrix_labels)
    time_points = np.linspace(0, full_matrix_labels[-1][1] + 180, config.time_evolution_steps)

    x, y, z = _build_query_grid(label_shape, affine, geometry, lesion_z_voxel, side_length)

    heatmaps, label_slices, volumes_3d, lesion_sizes = [], [], [], []
    for t in time_points:
        heatmaps.append(_predict_heatmap(model, t, x, y, z, side_length))
        volume, label_slice = _find_label_at_time(full_matrix_labels, t, lesion_z_voxel)
        volumes_3d.append(volume)
        label_slices.append(label_slice)
        lesion_sizes.append(float(volume.sum()))

    heatmaps = np.array(heatmaps)
    heatmap_peak = np.max(heatmaps, axis=(1, 2))
    heatmap_peak[heatmap_peak == 0] = 1.0
    heatmaps = heatmaps / heatmap_peak[:, np.newaxis, np.newaxis]

    volumes_3d = np.array(volumes_3d)
    label_slices = np.array(label_slices)
    lesion_sizes = np.array(lesion_sizes, dtype=float)

    voxel_volume_mm3 = np.prod(geometry.voxel_spacing)
    lesion_sizes_cm3 = lesion_sizes * voxel_volume_mm3 / 1000.0

    topdown_maps = [_build_topdown_height_map(volume) for volume in volumes_3d]

    change_points = np.zeros(len(time_points), dtype=bool)
    change_points[1:] = lesion_sizes_cm3[1:] != lesion_sizes_cm3[:-1]

    spacing_i, spacing_j = _heatmap_pixel_spacing(geometry, label_shape, side_length)
    heatmap_aspect = spacing_j / spacing_i

    return LesionTimeSeries(
        time_points=time_points,
        heatmaps=heatmaps,
        heatmap_aspect=heatmap_aspect,
        heatmap_pixel_spacing_mm=(spacing_i, spacing_j),
        label_slices=label_slices,
        volumes_3d=volumes_3d,
        topdown_maps=topdown_maps,
        lesion_sizes_cm3=lesion_sizes_cm3,
        change_points=change_points,
        lesion_z_voxel=lesion_z_voxel,
        geometry=geometry,
    )


def _build_topdown_height_map(volume: np.ndarray) -> np.ndarray:
    """Unchanged from train_plotting.py -- re-declared here only because it's
    a private (underscore-prefixed) name and `import *` doesn't pull those in."""
    height_map = np.full(volume.shape[:2], np.nan, dtype=float)
    lesion_x, lesion_y, lesion_z = np.where(volume == 1)
    for x_idx, y_idx, z_idx in zip(lesion_x, lesion_y, lesion_z):
        current = height_map[x_idx, y_idx]
        if np.isnan(current) or z_idx > current:
            height_map[x_idx, y_idx] = z_idx
    if np.isnan(height_map).all():
        return np.zeros(volume.shape[:2], dtype=float)
    return height_map


# ---------------------------------------------------------------------------
# Override: adapt to the patient (inner loop) before predicting/plotting
# ---------------------------------------------------------------------------

def plot_lesion_time_evolution(model, epoch, trj, config, final_side_length=False):
    """Render a multi-panel GIF of predicted lesion evolution vs. nnUNet
    ground-truth labels over time, and log it to mlflow.

    Overridden from train_plotting.py to do test-time adaptation first:
    builds a support set from the patient's own real scans and runs
    `config.inner_steps` SGD steps (mirrors `validate_polation` in
    train_meta.py) before querying the model, so the GIF reflects the
    patient-adapted weights instead of the raw shared meta-init.

    The adaptation + all model queries happen inside the same
    `higher.innerloop_ctx` block -- the fast-weights model (`fmodel`) isn't
    guaranteed usable once that context exits, so `_assemble_time_series`
    is called from inside it rather than after.

    Wrapped in `torch.enable_grad()`: train_meta.py currently calls this
    function from inside a `with torch.no_grad():` block. `enable_grad()`
    locally overrides that so the inner loop can backprop; harmless if you
    remove that outer no_grad wrapper later.
    """
    side_length = config.full_size_side_length if final_side_length else config.time_evolution_side_length
    patient_idx = trj.embedding_id  # display/filename use only -- no longer fed into the model

    full_matrix_labels, affine = _load_full_matrix_with_affine(trj)
    label_shape = np.array(full_matrix_labels[0][0].shape)  # (X, Y, Z)
    geometry = _compute_reference_geometry(label_shape, affine)

    coords_np, labels_np = _build_support_set_from_labels(full_matrix_labels, affine, geometry, config)
    supp_coords = torch.from_numpy(coords_np).float().unsqueeze(0).to(config.device)
    supp_labels = torch.from_numpy(labels_np).float().unsqueeze(0).unsqueeze(-1).to(config.device)

    inner_lr = getattr(config, 'inner_lr', 0.01)
    inner_steps = getattr(config, 'inner_steps', 3)
    criterion = config.loss_fn

    model.to(config.device)
    was_training = model.training
    model.eval()

    with torch.enable_grad():
        inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
        with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
            for _ in range(inner_steps):
                supp_preds = fmodel(supp_coords)
                inner_loss = criterion(supp_preds, supp_labels)
                diffopt.step(inner_loss)

            # Everything that queries the model happens while fmodel is still valid.
            data = _assemble_time_series(
                fmodel, config, full_matrix_labels, label_shape, affine, geometry, side_length
            )

    if was_training:
        model.train()

    artists = _build_figure(data, patient_idx)

    def update(frame_idx):
        return _update_frame(frame_idx, data, artists, patient_idx)

    output_path = (
        f"/tmp/{epoch:04d}_epoch_lesion_heatmap_patient_"
        f"{trj.patient_id}_{patient_idx}_lesion_{trj.label_id}.gif"
    )
    _save_gif(artists.fig, update, len(data.heatmaps), output_path)

    mlflow.log_artifact(output_path)
    plt.close(artists.fig)