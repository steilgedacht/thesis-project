import torch
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from scipy import ndimage
from .patient import Patient
from dataclasses import dataclass
from typing import List, Tuple


def _get_voxel_spacing(affine):
    """Physical voxel spacing (mm) per axis, from a 4x4 NIfTI-style affine."""
    return np.abs(np.diag(affine)[:3])

def _resize_2d(img, target_shape, order=0):
    """Resize a 2D array to an exact target shape (order=0 keeps masks binary)."""
    if img.shape == tuple(target_shape):
        return img
    zoom_factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return ndimage.zoom(img, zoom_factors, order=order)

def _nice_scale_length_mm(pixel_extent_mm):
    """Pick a round physical length for a scale bar, roughly 1/4 of the visible width."""
    target = pixel_extent_mm / 4
    magnitude = 10 ** np.floor(np.log10(target))
    for mult in (1, 2, 2.5, 5, 10):
        candidate = mult * magnitude
        if candidate >= target:
            return candidate
    return 10 * magnitude

def _is_lstm_model(model):
    """Detect if model is LesionLSTM/LesionLatentLSTM vs INR."""
    model_class_name = model.__class__.__name__
    return 'LSTM' in model_class_name or 'Latent' in model_class_name

def plot_predictions(v_preds, v_labels, v_coords, epoch, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Get 3D prediction volume by reshaping predictions
    pred_volume = v_preds[0].squeeze().detach().cpu().numpy()
    label_volume = v_labels[0].squeeze().detach().cpu().numpy()
    pred_coords = v_coords[0,:,:3].squeeze().detach().cpu().numpy()

    # Top view (XY plane, max projection along Z)
    axes[1,0].scatter(pred_coords[:,0], pred_coords[:,1], c=pred_volume, s=5, alpha=0.5, cmap='copper', vmin=0, vmax=1)
    axes[1,0].set_title('Prediction - Top View')
    axes[1,0].axis('off')

    # Side view (XZ plane, max projection along Y)
    axes[0,0].scatter(pred_coords[:,0], pred_coords[:,2], c=pred_volume, s=5, alpha=0.5, cmap='copper', vmin=0, vmax=1)
    axes[0,0].set_title('Prediction - Front View')
    axes[0,0].axis('off')

    axes[0,1].scatter(pred_coords[:,1], pred_coords[:,2], c=pred_volume, s=5, alpha=0.5, cmap='copper', vmin=0, vmax=1)
    axes[0,1].set_title('Prediction - Side View')
    axes[0,1].axis('off')

    axes[1,1].scatter(pred_coords[:,0], pred_coords[:,1], c=label_volume, s=5, alpha=0.5, cmap='copper', vmin=0, vmax=1)
    axes[1,1].set_title('Labels')
    axes[1,1].axis('off')

    plt.tight_layout()
    mlflow.log_figure(fig, f"{epoch:04d}_epoch_predictions_{title}.png")
    plt.close(fig)

def plot_heatmap(model, labels, coords, patient_idx, epoch, sample_id, config, time_point=None):
    # get the height where the lesion is located
    center_of_mass = torch.mean(coords.squeeze()[:len(labels.squeeze())//2,:3], dim=0).detach().cpu().numpy()
    lesion_z = center_of_mass[2]

    if time_point is not None:
        sample_id = str(sample_id) + "_time_point_" + str(time_point)
    
    meshgrid = np.meshgrid(np.linspace(0, 1, config.full_size_side_length), np.linspace(0, 1, config.full_size_side_length), indexing='ij')
    x = meshgrid[0].flatten() * 2 - 1
    y = meshgrid[1].flatten() * 2 - 1
    
    z = np.full_like(x, lesion_z)
    if time_point is None:
        time_point = np.full_like(x, coords[0,0,-1].detach().cpu().numpy()) 
    else:
        time_point = np.full_like(x, time_point) 

    hm_coords = np.stack([x, y, z, time_point], axis=1)
    coords_tensor = torch.from_numpy(hm_coords).float().to(next(model.parameters()).device).unsqueeze(0)

    with torch.no_grad():
        for slice_idx in range(0, coords_tensor.shape[1], 1000):
            slice_coords = coords_tensor[:, slice_idx:slice_idx+1000]
            slice_preds = model(slice_coords, patient_idx)
            if slice_idx == 0:
                heatmap_preds = slice_preds.cpu().numpy()
            else:
                heatmap_preds = np.concatenate([heatmap_preds, slice_preds.cpu().numpy()], axis=1)

    heatmap_preds = heatmap_preds.reshape(config.full_size_side_length, config.full_size_side_length)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap_preds, cmap='copper', vmin=0, vmax=1)
    ax.set_title(f'Predicted Lesion Heatmap {patient_idx.item()}')
    ax.axis('off')
    mlflow.log_figure(fig, f"{epoch:04d}_epoch_lesion_heatmap_patient_{patient_idx.item()}_sample_{sample_id}.png")
    plt.close(fig)
    
def visualize_samples(model, epoch, monitoring_samples, data_loader, text, config):
    for sample_idx in monitoring_samples:
        coords, labels, patient_idx = data_loader.dataset[sample_idx]
        coords = torch.from_numpy(coords).float().unsqueeze(0).to(config.device)
        labels = torch.from_numpy(labels).float().unsqueeze(0).unsqueeze(-1).to(config.device)
        patient_idx = torch.tensor(patient_idx).unsqueeze(0).to(config.device)
        predictions = model(coords, patient_idx)
        plot_predictions(predictions, labels, coords, epoch, f"{text}_patient_{patient_idx.item()}_sample_{sample_idx}")



@dataclass
class ReferenceGeometry:
    """Physical-space reference info, all derived from the real label-grid
    affine (the same affine `LesionDataset.prepare_data` uses)."""
    affine: np.ndarray
    volume_center_phys: np.ndarray  # mm, grid's geometric center -- training's origin
    voxel_spacing: np.ndarray       # mm per voxel along (i, j, k)
    label_aspect: float             # voxel_spacing[1] / voxel_spacing[0]
 
 
@dataclass
class LesionTimeSeries:
    """Everything the animation needs, pre-computed once."""
    time_points: np.ndarray
    heatmaps: np.ndarray            # (T, S, S), normalized to [0, 1] per frame
    heatmap_aspect: float           # aspect ratio for the S x S query grid specifically
    heatmap_pixel_spacing_mm: Tuple[float, float]  # (row, col) mm per heatmap pixel
    label_slices: np.ndarray        # (T, X, Y) 2D label slice at lesion height
    volumes_3d: np.ndarray          # (T, X, Y, Z) full label volumes
    topdown_maps: List[np.ndarray]  # (T,) each (X, Y) height map
    lesion_sizes_cm3: np.ndarray    # (T,)
    change_points: np.ndarray       # (T,) bool, True where lesion size changed
    lesion_z_voxel: float           # trimmed-mean lesion z, in voxel index units
    geometry: ReferenceGeometry
 
 
@dataclass
class AnimationArtists:
    """Matplotlib objects the per-frame update function needs to mutate."""
    fig: object
    ax_heatmap: object
    im_heatmap: object
    scale_line: object
    scale_text: object
    im_label: object
    scatter_3d: object
    cube_scale: Tuple[float, float, float]
    im_overlap_heatmap: object
    im_overlap_label: object
    progress_line: object
    current_marker: object
    mid_y: float
    im_topdown: object
 
 
# ---------------------------------------------------------------------------
# Step 1: geometry / reference frame (matches LesionDataset.prepare_data)
# ---------------------------------------------------------------------------
 
def _load_full_matrix_with_affine(trj):
    """Load every timepoint's label volume together with the physical affine.
 
    ASSUMPTION: mirrors the per-date training call
    `(labels, day_number), affine = trj.load_labels_for_inr(selected_date=..., affine=True)`,
    just batched across all dates and without `selected_date`. Verify this
    matches your actual `load_labels_for_inr` signature -- if it returns a
    separate affine per timepoint instead of one shared affine, this
    function and `_assemble_time_series` need to carry a list of affines.
    """
    full_matrix_labels, affine = trj.load_labels_for_inr(absolute_day_number=True, affine=True)
    return full_matrix_labels, affine
 
 
def _compute_lesion_z_voxel(full_matrix_labels) -> float:
    """Trimmed-mean z *voxel index* of the lesion. Used only to choose which
    axial slice to display -- the model itself is queried with this z voxel
    run through the full affine transform in `_voxel_to_model_space`, not
    with this raw index directly."""
    centers_of_mass_z = np.stack([
        ndimage.center_of_mass(labels)[-1] for labels, _ in full_matrix_labels
    ])
    return float(np.mean(np.sort(centers_of_mass_z)[1:-1]))
 
 
def _volume_center_phys(label_shape: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Geometric center of the voxel grid in physical (mm) space -- the same
    reference point `LesionDataset.prepare_data` centers training coords on."""
    center_voxel = (np.asarray(label_shape) - 1) / 2.0
    return affine[:3, :3] @ center_voxel + affine[:3, 3]
 
 
def _voxel_to_model_space(voxel_ijk: np.ndarray, affine: np.ndarray,
                           volume_center_phys: np.ndarray) -> np.ndarray:
    """Voxel indices (N, 3) -> the exact coordinate space the INR was trained
    on. Mirrors `LesionDataset.prepare_data` exactly: affine to physical mm,
    center on the grid's geometric center, scale so 1 unit = 10 cm."""
    ones = np.ones((voxel_ijk.shape[0], 1))
    homogeneous = np.concatenate([voxel_ijk, ones], axis=1)
    physical = (affine @ homogeneous.T).T[:, :3]
    return (physical - volume_center_phys) / 100.0
 
 
def _compute_reference_geometry(label_shape: np.ndarray, affine: np.ndarray) -> ReferenceGeometry:
    voxel_spacing = _get_voxel_spacing(affine)
    label_aspect = voxel_spacing[1] / voxel_spacing[0]
    volume_center_phys = _volume_center_phys(label_shape, affine)
    return ReferenceGeometry(affine, volume_center_phys, voxel_spacing, label_aspect)
 
 
def _heatmap_pixel_spacing(geometry: ReferenceGeometry, label_shape: np.ndarray,
                            side_length: int) -> Tuple[float, float]:
    """mm spanned by one heatmap pixel along (row, col). The query grid
    resamples the label grid's full x/y physical extent at `side_length`
    resolution, so its per-pixel spacing differs from the label grid's
    native voxel spacing by the resampling ratio on each axis."""
    spacing_i = geometry.voxel_spacing[0] * (label_shape[0] - 1) / max(side_length - 1, 1)
    spacing_j = geometry.voxel_spacing[1] * (label_shape[1] - 1) / max(side_length - 1, 1)
    return spacing_i, spacing_j
 
 
# ---------------------------------------------------------------------------
# Step 2: model inference + label matching over time
# ---------------------------------------------------------------------------

def _predict_heatmap_inr(model, patient_idx_tensor, t, x, y, z, side_length, batch_size=150_000):
    """Query the INR for one timepoint, in chunks to bound memory use."""
    time_col = np.full_like(x, t)
    coords = np.stack([x, y, z, time_col], axis=1)
    coords_tensor = torch.from_numpy(coords).float().to(next(model.parameters()).device).unsqueeze(0)

    chunks = []
    with torch.no_grad():
        for start in range(0, coords_tensor.shape[1], batch_size):
            chunk = coords_tensor[:, start:start + batch_size]
            chunks.append(model(chunk, patient_idx_tensor).cpu().numpy())
    preds = np.concatenate(chunks, axis=1)
    return preds.reshape(side_length, side_length)


def _extract_2d_slice_from_grid(grid_3d, lesion_z_voxel):
    """Extract a 2D axial slice at lesion height from a 3D grid.
    
    Args:
        grid_3d: [D, H, W] or [1, D, H, W] tensor or array
        lesion_z_voxel: z index for slicing
        
    Returns:
        2D slice [H, W]
    """
    if isinstance(grid_3d, torch.Tensor):
        grid_3d = grid_3d.cpu().numpy()
    
    # Handle batch dimension if present
    if grid_3d.ndim == 4:
        grid_3d = grid_3d[0]  # Take first batch
    elif grid_3d.ndim == 5:  # [B, T, 1, D, H, W] - take last time step
        grid_3d = grid_3d[0, -1, 0]
    elif grid_3d.ndim == 3 and grid_3d.shape[0] == 1:
        grid_3d = grid_3d[0]
        
    # Extract z slice
    z_size = grid_3d.shape[0] if grid_3d.ndim == 3 else grid_3d.shape[-1]
    z_idx = int(np.clip(round(lesion_z_voxel), 0, z_size - 1))
    
    if grid_3d.ndim == 3:
        return grid_3d[z_idx]
    else:
        return grid_3d[..., z_idx]


def _predict_grids_lstm(model, patient_idx_tensor, config, time_points, observed_grids, 
                        observed_times, lesion_z_voxel):
    """Predict LSTM grids over time using autoregressive rollout.
    
    For LSTM: runs one forward pass per time point, extracting 2D slices.
    Much more memory-efficient than coordinate queries.
    
    Args:
        model: LesionLSTM or LesionLatentLSTM
        patient_idx_tensor: [1] patient ID
        config: config object with device
        time_points: array of target times to predict at
        observed_grids: [1, T_obs, 1, D, H, W] observed history
        observed_times: [1, T_obs] observed timestamps
        lesion_z_voxel: lesion z position for 2D slice extraction
        
    Returns:
        heatmaps: [T, S, S] 2D slices at each time point
    """
    device = config.device
    model.eval()
    heatmaps = []
    
    with torch.no_grad():
        for t in time_points:
            # Build time array: observed times + target time
            full_times = torch.cat([
                observed_times,
                torch.tensor([[t]], dtype=torch.float32, device=device)
            ], dim=1)
            
            # Run model with teacher forcing on history, predict one step ahead
            preds = model(
                observed_grids.to(device),
                full_times.to(device),
                patient_idx_tensor.to(device),
                n_future=1
            )  # [1, T_obs, 1, D, H, W]
            
            # Extract the last predicted step (the future prediction)
            last_pred = preds[:, -1]  # [1, 1, D, H, W]
            
            # Apply sigmoid to convert logits to probabilities
            last_pred = torch.sigmoid(last_pred)
            
            # Extract 2D slice at lesion height
            slice_2d = _extract_2d_slice_from_grid(last_pred, lesion_z_voxel)
            heatmaps.append(slice_2d)
            
            # Clear GPU memory
            del preds, last_pred
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    heatmaps = np.array(heatmaps)
    return heatmaps


def _predict_heatmap(model, patient_idx_tensor, t, x, y, z, side_length, batch_size=150_000):
    """Legacy function for compatibility - detects model type and routes accordingly."""
    if _is_lstm_model(model):
        raise ValueError("Use _predict_grids_lstm for LSTM models, not _predict_heatmap")
    return _predict_heatmap_inr(model, patient_idx_tensor, t, x, y, z, side_length, batch_size)
 
 
def _find_label_at_time(full_matrix_labels, t, lesion_z_voxel):
    """Most recent label (day <= t), plus its 2D slice at lesion height."""
    for volume, day in reversed(full_matrix_labels):
        if day <= t:
            z_size = volume.shape[-1]
            z_idx = int(np.clip(round(lesion_z_voxel), 0, z_size - 1))
            return volume, volume[:, :, z_idx]
    raise ValueError(f"No label found at or before t={t}.")
 
 
def _build_topdown_height_map(volume: np.ndarray) -> np.ndarray:
    """Max lesion-voxel z-index per (x, y) column; columns with no lesion -> 0."""
    height_map = np.full(volume.shape[:2], np.nan, dtype=float)
    lesion_x, lesion_y, lesion_z = np.where(volume == 1)
    for x_idx, y_idx, z_idx in zip(lesion_x, lesion_y, lesion_z):
        current = height_map[x_idx, y_idx]
        if np.isnan(current) or z_idx > current:
            height_map[x_idx, y_idx] = z_idx
    if np.isnan(height_map).all():
        return np.zeros(volume.shape[:2], dtype=float)
    return height_map
 
 
def _build_query_grid(label_shape: np.ndarray, affine: np.ndarray, geometry: ReferenceGeometry,
                       lesion_z_voxel: float, side_length: int):
    """Voxel indices spanning the label grid's full x/y extent at
    `side_length` resolution, held at the lesion's z voxel -- then converted
    to the model's training coordinate space."""
    grid_i, grid_j = np.meshgrid(
        np.linspace(0, label_shape[0] - 1, side_length),
        np.linspace(0, label_shape[1] - 1, side_length),
        indexing='ij',
    )
    voxel_ijk = np.stack([
        grid_i.flatten(), grid_j.flatten(), np.full(grid_i.size, lesion_z_voxel),
    ], axis=1)
    xyz = _voxel_to_model_space(voxel_ijk, affine, geometry.volume_center_phys)
    return xyz[:, 0], xyz[:, 1], xyz[:, 2]
 
 
def _assemble_time_series(model, patient_idx_tensor, config, full_matrix_labels,
                           label_shape, affine, geometry, side_length) -> LesionTimeSeries:
    """Assemble time series, routing to INR or LSTM implementation based on model type."""
    if _is_lstm_model(model):
        return _assemble_time_series_lstm(
            model, patient_idx_tensor, config, full_matrix_labels, 
            label_shape, affine, geometry, side_length
        )
    else:
        return _assemble_time_series_inr(
            model, patient_idx_tensor, config, full_matrix_labels,
            label_shape, affine, geometry, side_length
        )


def _assemble_time_series_lstm(model, patient_idx_tensor, config, full_matrix_labels,
                                label_shape, affine, geometry, side_length) -> LesionTimeSeries:
    """Assemble time series for LSTM by loading observation grids and predicting forward.
    
    MEMORY EFFICIENT: One forward pass per time point, extracts 2D slices directly.
    No coordinate queries needed.
    """
    from .dataloader_lstm import LesionSequenceDataset
    from .mri_dataloader import MRI_Dataloader
    
    lesion_z_voxel = _compute_lesion_z_voxel(full_matrix_labels)
    
    # Use time points from labels (observation times + extrapolation into future)
    label_times = np.array([day for _, day in full_matrix_labels])
    last_time = label_times[-1]
    future_days = config.time_evolution_steps - len(label_times)
    max_future = min(last_time + 365, last_time * 1.5)  # Don't predict too far
    
    if future_days > 0:
        future_times = np.linspace(last_time + 1, max_future, future_days)
        time_points = np.concatenate([label_times, future_times])
    else:
        time_points = label_times[:config.time_evolution_steps]
    
    # Load observed grids (all labels)
    observed_grids_list = []
    observed_times_list = []
    
    for labels, day in full_matrix_labels[:config.max_sequence_len]:  # Use history up to max sequence length
        # Resize label to LSTM grid size
        grid = ndimage.zoom(labels.astype(float), 
                           np.array(config.lstm_grid_size) / np.array(labels.shape),
                           order=1)
        observed_grids_list.append(torch.from_numpy(grid[None, None]).float())  # [1, D, H, W]
        observed_times_list.append(day)
    
    observed_grids = torch.stack(observed_grids_list, dim=1)  # [1, T, 1, D, H, W]
    observed_times = torch.tensor([observed_times_list], dtype=torch.float32)  # [1, T]
    
    # Predict grids over time
    heatmaps = _predict_grids_lstm(model, patient_idx_tensor, config, 
                                   time_points, observed_grids, observed_times, 
                                   lesion_z_voxel)
    
    # Normalize heatmaps
    heatmap_peak = np.max(heatmaps, axis=(1, 2), keepdims=True)
    heatmap_peak[heatmap_peak == 0] = 1.0
    heatmaps = heatmaps / heatmap_peak
    
    # Get labels at each time point
    label_slices, volumes_3d, lesion_sizes = [], [], []
    for t in time_points:
        volume, label_slice = _find_label_at_time(full_matrix_labels, t, lesion_z_voxel)
        volumes_3d.append(volume)
        label_slices.append(label_slice)
        lesion_sizes.append(float(volume.sum()))
    
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


def _assemble_time_series_inr(model, patient_idx_tensor, config, full_matrix_labels,
                               label_shape, affine, geometry, side_length) -> LesionTimeSeries:
    """Original INR-based time series assembly using coordinate queries."""
    lesion_z_voxel = _compute_lesion_z_voxel(full_matrix_labels)
    time_points = np.linspace(0, full_matrix_labels[-1][1] + 180, config.time_evolution_steps)

    x, y, z = _build_query_grid(label_shape, affine, geometry, lesion_z_voxel, side_length)

    heatmaps, label_slices, volumes_3d, lesion_sizes = [], [], [], []
    for t in time_points:
        heatmaps.append(_predict_heatmap_inr(model, patient_idx_tensor, t, x, y, z, side_length))
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
 
 
# ---------------------------------------------------------------------------
# Step 3: figure construction (one builder per panel)
# ---------------------------------------------------------------------------
 
def _init_heatmap_panel(ax, data: LesionTimeSeries, patient_idx):
    im = ax.imshow(data.heatmaps[0], cmap='copper', vmin=0, vmax=1,
                    aspect=data.heatmap_aspect, animated=True)
    ax.set_title(f"Predicted Lesion Heatmap {patient_idx} frame 0/{len(data.heatmaps)}")
    ax.axis('off')
 
    img_h_px, img_w_px = data.heatmaps[0].shape
    _, col_spacing = data.heatmap_pixel_spacing_mm  # mm per heatmap pixel, not label-grid spacing
    bar_length_mm = _nice_scale_length_mm(img_w_px * col_spacing)
    bar_length_px = bar_length_mm / col_spacing
 
    bar_x0, bar_y = img_w_px * 0.05, img_h_px * 0.92
    scale_line, = ax.plot([bar_x0, bar_x0 + bar_length_px], [bar_y, bar_y],
                           color='white', lw=3, solid_capstyle='butt')
    scale_text = ax.text(bar_x0 + bar_length_px / 2, bar_y - img_h_px * 0.03,
                          f"{bar_length_mm:.0f} mm", color='white', ha='center', va='bottom',
                          fontsize=10)
    return im, scale_line, scale_text
 
 
def _init_label_panel(ax, data: LesionTimeSeries):
    im = ax.imshow(data.label_slices[0], cmap='copper', vmin=0, vmax=1, aspect=data.geometry.label_aspect)
    ax.set_title("Lesion Label of nnUNet")
    ax.axis('off')
    return im
 
 
def _init_3d_panel(ax, data: LesionTimeSeries):
    nx, ny, nz = data.volumes_3d[0].shape
    cube_extent = max(nx, ny, nz, 1)
    scale = (cube_extent / max(nx, 1), cube_extent / max(ny, 1), cube_extent / max(nz, 1))
 
    xs, ys, zs = np.where(data.volumes_3d[0] == 1)
    scatter = ax.scatter(xs * scale[0], ys * scale[1], zs * scale[2])
 
    ax.set_title("Lesion Label in 3D space")
    ax.set_xlim3d(0, cube_extent)
    ax.set_ylim3d(0, cube_extent)
    ax.set_zlim3d(0, cube_extent)

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.zaxis.set_tick_params(labelbottom=False)

    ax.set_box_aspect((1, 1, 1))
 
    plane_x, plane_y = np.meshgrid(np.linspace(0, cube_extent, 10), np.linspace(0, cube_extent, 10))
    z_slice_idx = int(np.clip(round(data.lesion_z_voxel), 0, nz - 1))
    plane_z = np.full_like(plane_x, z_slice_idx * scale[2])
    ax.plot_surface(plane_x, plane_y, plane_z, alpha=0.3, color='lightblue',
                     antialiased=False, label="slice_of_heatmap")
    
    return scatter, scale
 
 
def _init_overlap_panel(ax, data: LesionTimeSeries):
    heatmap0_resized = _resize_2d(data.heatmaps[0], data.label_slices[0].shape)
    im_heatmap = ax.imshow(heatmap0_resized, cmap='copper', vmin=0, vmax=1,
                            aspect=data.geometry.label_aspect, animated=True)
    im_label = ax.imshow(data.label_slices[0], cmap='Reds', vmin=0, vmax=1, alpha=0.5,
                          aspect=data.geometry.label_aspect)
    ax.set_title("Lesion Label + Prediction Overlap")
    ax.axis('off')
    return im_heatmap, im_label
 
 
def _init_timeline_panel(ax, data: LesionTimeSeries):
    max_vol = max(data.lesion_sizes_cm3.max(), 1e-6)
    mid_y = max_vol / 2
 
    ax.fill_between(data.time_points, 0, data.lesion_sizes_cm3, color='skyblue', alpha=0.25)
    ax.plot(data.time_points, data.lesion_sizes_cm3, color='steelblue', lw=1.5, label='Lesion size [cm³]')
    ax.plot(data.time_points, np.full_like(data.time_points, mid_y), color='lightgray', lw=1)
    ax.scatter(data.time_points[data.change_points],
               np.full(data.change_points.sum(), mid_y), marker='|', color='red', s=120, label='Label Timepoints')
 
    progress_line, = ax.plot(
        [data.time_points[0], data.time_points[0]],
        [mid_y - max_vol * 0.25, mid_y + max_vol * 0.25],
        color='royalblue', lw=3,
    )
    current_marker, = ax.plot([data.time_points[0]], [mid_y], marker='o', color='royalblue', markersize=8, label='Current position')
 
    ax.set_xlim(data.time_points[0], data.time_points[-1])
    ax.set_ylim(-max_vol * 0.1, max_vol * 1.2)
    ax.set_yticks(np.linspace(0, max_vol, 4))
    ax.set_xticks(np.linspace(data.time_points[0], data.time_points[-1], 5))
    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Lesion Size [cm³]')
    ax.set_title('Lesion Growth Time Evolution Timeline')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    return progress_line, current_marker, mid_y
 
 
def _init_topdown_panel(ax, data: LesionTimeSeries):
    nz = data.volumes_3d[0].shape[-1]
    topdown_max = max(1, nz - 1)
    im = ax.imshow(data.topdown_maps[0], cmap='terrain', vmin=0, vmax=topdown_max,
                    aspect=data.geometry.label_aspect)
    ax.set_title('Top-down lesion height map')
    ax.axis('off')

    # Add colorbar on the left
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.yaxis.tick_right()
    return im
 
 
def _build_figure(data: LesionTimeSeries, patient_idx) -> AnimationArtists:
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
 
    ax_heatmap = fig.add_subplot(gs[0, 0])
    im_heatmap, scale_line, scale_text = _init_heatmap_panel(ax_heatmap, data, patient_idx)
 
    ax_label = fig.add_subplot(gs[0, 1])
    im_label = _init_label_panel(ax_label, data)
 
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
    scatter_3d, cube_scale = _init_3d_panel(ax_3d, data)
 
    ax_overlap = fig.add_subplot(gs[0, 2])
    im_overlap_heatmap, im_overlap_label = _init_overlap_panel(ax_overlap, data)
 
    ax_timeline = fig.add_subplot(gs[1, 2])
    progress_line, current_marker, mid_y = _init_timeline_panel(ax_timeline, data)
 
    ax_topdown = fig.add_subplot(gs[1, 1])
    im_topdown = _init_topdown_panel(ax_topdown, data)
 
    fig.tight_layout()
 
    return AnimationArtists(
        fig=fig,
        ax_heatmap=ax_heatmap, im_heatmap=im_heatmap, scale_line=scale_line, scale_text=scale_text,
        im_label=im_label,
        scatter_3d=scatter_3d, cube_scale=cube_scale,
        im_overlap_heatmap=im_overlap_heatmap, im_overlap_label=im_overlap_label,
        progress_line=progress_line, current_marker=current_marker, mid_y=mid_y,
        im_topdown=im_topdown,
    )
 
 
# ---------------------------------------------------------------------------
# Step 4: per-frame update + GIF export
# ---------------------------------------------------------------------------
 
def _update_frame(frame_idx, data: LesionTimeSeries, artists: AnimationArtists, patient_idx):
    is_last_frame = frame_idx == len(data.heatmaps) - 1
 
    if is_last_frame:
        artists.im_heatmap.set_array(np.ones_like(data.heatmaps[0]))
        artists.im_label.set_array(np.ones_like(data.label_slices[0]))
        artists.im_overlap_heatmap.set_array(np.ones_like(artists.im_overlap_heatmap.get_array()))
        artists.im_overlap_label.set_array(np.ones_like(data.label_slices[0]))
        artists.scatter_3d._offsets3d = ([], [], [])
        artists.ax_heatmap.set_title("--- End of Sequence ---")
        artists.progress_line.set_data([data.time_points[0], data.time_points[-1]],
                                        [artists.mid_y, artists.mid_y])
        artists.current_marker.set_data([data.time_points[-1]], [artists.mid_y])
        artists.im_topdown.set_array(data.topdown_maps[-1])
    else:
        artists.im_heatmap.set_array(data.heatmaps[frame_idx])
        artists.im_label.set_array(data.label_slices[frame_idx])
        artists.im_overlap_heatmap.set_array(
            _resize_2d(data.heatmaps[frame_idx], data.label_slices[frame_idx].shape)
        )
        artists.im_overlap_label.set_array(data.label_slices[frame_idx])
 
        sx, sy, sz = artists.cube_scale
        xs, ys, zs = np.where(data.volumes_3d[frame_idx] == 1)
        artists.scatter_3d._offsets3d = (xs * sx, ys * sy, zs * sz)
 
        artists.ax_heatmap.set_title(
            f'Predicted Lesion Heatmap {patient_idx} frame {frame_idx:03d}/{len(data.heatmaps)}'
        )
        artists.progress_line.set_data([data.time_points[0], data.time_points[frame_idx]],
                                        [artists.mid_y, artists.mid_y])
        artists.current_marker.set_data([data.time_points[frame_idx]], [artists.mid_y])
        artists.im_topdown.set_array(data.topdown_maps[frame_idx])
 
    return [
        artists.im_heatmap, artists.im_label, artists.im_overlap_heatmap, artists.im_overlap_label,
        artists.scatter_3d, artists.progress_line, artists.current_marker, artists.im_topdown,
        artists.scale_line, artists.scale_text, artists.ax_heatmap.title,
    ]
 
 
def _save_gif(fig, update_fn, n_frames: int, output_path: str):
    """Render and save the animation, tolerating either ImageMagick binary name."""
    ani = FuncAnimation(fig, update_fn, frames=n_frames, interval=50)
    last_error = None
    for convert_path in ('convert', 'magick'):
        try:
            plt.rcParams['animation.convert_path'] = convert_path
            writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
            ani.save(output_path, writer=writer, dpi=50)
            return
        except Exception as exc:  # noqa: BLE001 - genuinely need to try the other binary name
            last_error = exc
    raise last_error
 
 
# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
 
def plot_lesion_time_evolution(model, epoch, trj, config, final_side_length=False):
    """Render a multi-panel GIF of predicted lesion evolution vs. nnUNet
    ground-truth labels over time, and log it to mlflow.
 
    Panels: predicted heatmap, ground-truth label slice, 3D label scatter,
    heatmap/label overlap, lesion-size timeline, top-down height map.
    """
    side_length = config.full_size_side_length if final_side_length else config.time_evolution_side_length
    patient_idx_tensor = torch.tensor(trj.embedding_id).unsqueeze(0).to(config.device)
    patient_idx = patient_idx_tensor.item()
 
    full_matrix_labels, affine = _load_full_matrix_with_affine(trj)
    label_shape = np.array(full_matrix_labels[0][0].shape)  # (X, Y, Z)
 
    geometry = _compute_reference_geometry(label_shape, affine)
    data = _assemble_time_series(
        model, patient_idx_tensor, config, full_matrix_labels, label_shape, affine, geometry, side_length
    )
 
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
 
