import torch
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from scipy import ndimage


def _get_voxel_spacing(affine):
    """Physical voxel spacing (mm) per axis from a 4x4 NIfTI-style affine."""
    if affine is None:
        return np.array([1.0, 1.0, 1.0])
    affine = np.asarray(affine, dtype=float)
    if affine.shape == (4, 4):
        return np.abs(np.diag(affine)[:3])
    if affine.shape == (3, 3):
        return np.abs(np.diag(affine)[:3])
    return np.array([1.0, 1.0, 1.0])


def _resize_2d(img, target_shape, order=0):
    """Resize a 2D array to an exact target shape (order=0 keeps masks binary)."""
    if img.shape == tuple(target_shape):
        return img.astype(float, copy=False)
    zoom_factors = (target_shape[0] / img.shape[0], target_shape[1] / img.shape[1])
    return ndimage.zoom(img, zoom_factors, order=order)


def _nice_scale_length_mm(pixel_extent_mm):
    """Pick a round physical length for a scale bar, roughly 1/4 of the visible width."""
    target = pixel_extent_mm / 4
    magnitude = 10 ** np.floor(np.log10(target))
    for mult in (1, 2, 2.5, 5, 10):
        if target / magnitude <= mult:
            return mult * magnitude
    return 10 * magnitude

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


def plot_lesion_time_evolution(model, epoch, trj, config, final_side_length=False):
    side_length = config.time_evolution_side_length if not final_side_length else config.full_size_side_length
    patient_idx = torch.tensor(trj.embedding_id).unsqueeze(0).to(config.device)

    full_matrix_labels = trj.load_labels_for_inr(absolute_day_number=True)
    if not full_matrix_labels:
        return

    label_shape = np.array(full_matrix_labels[0][0].shape)
    label_affine = np.diag([max(1.0, label_shape[0]), max(1.0, label_shape[1]), max(1.0, label_shape[2]), 1.0])
    label_spacing = _get_voxel_spacing(label_affine)
    label_aspect = label_spacing[1] / label_spacing[0]

    heatmaps = []
    heatmap_affines = []
    labels_list = []
    list_3d = []
    lesion_sizes = []

    with torch.no_grad():
        center_of_mass = np.stack([
            ndimage.center_of_mass(volume)[-1] for volume, _ in full_matrix_labels if np.any(volume)
        ])
        if center_of_mass.size > 0:
            avg_center_of_mass = float(np.mean(np.sort(center_of_mass)[1:-1])) if center_of_mass.size > 2 else float(np.mean(center_of_mass))
        else:
            avg_center_of_mass = 0.0
        lesion_z_norm = avg_center_of_mass / max(label_shape[-1] - 1, 1) * 2 - 1

        meshgrid = np.meshgrid(np.linspace(0, 1, side_length), np.linspace(0, 1, side_length), indexing='ij')
        x = meshgrid[0].flatten() * 2 - 1
        y = meshgrid[1].flatten() * 2 - 1
        z = np.full_like(x, lesion_z_norm)

        time_points = np.linspace(0, full_matrix_labels[-1][1] + 180, config.time_evolution_steps)

        for t in time_points:
            time_point = np.full_like(x, t)
            hm_coords = np.stack([x, y, z, time_point], axis=1)
            coords_tensor = torch.from_numpy(hm_coords).float().to(next(model.parameters()).device).unsqueeze(0)

            for slice_idx in range(0, coords_tensor.shape[1], 150_000):
                slice_coords = coords_tensor[:, slice_idx:slice_idx + 150_000]
                slice_preds = model(slice_coords, patient_idx)
                if slice_idx == 0:
                    heatmap_preds = slice_preds.cpu().numpy()
                else:
                    heatmap_preds = np.concatenate([heatmap_preds, slice_preds.cpu().numpy()], axis=1)

            heatmap_preds = heatmap_preds.reshape(side_length, side_length)
            heatmaps.append(heatmap_preds)
            heatmap_affines.append(label_affine)

            for element in reversed(full_matrix_labels):
                if element[1] <= t:
                    z_size = element[0].shape[-1]
                    z_idx = int(np.clip(((lesion_z_norm + 1) / 2) * (z_size - 1), 0, z_size - 1))
                    labels_list.append(element[0][:, :, z_idx])
                    list_3d.append(element[0])
                    lesion_sizes.append(float(element[0].sum()))
                    break

    heatmaps = np.array(heatmaps)
    heatmap_max = np.max(heatmaps, axis=(1, 2))
    heatmap_max[heatmap_max == 0] = 1.0
    heatmaps = heatmaps / heatmap_max[:, np.newaxis, np.newaxis]

    label_grid = np.array(labels_list)
    list_3d = np.array(list_3d)
    lesion_sizes = np.array(lesion_sizes, dtype=float)
    if lesion_sizes.size > 0:
        lesion_sizes = lesion_sizes / max(lesion_sizes.max(), 1.0)

    def build_topdown_height(volume):
        height_map = np.full(volume.shape[:2], np.nan, dtype=float)
        for x_idx in range(volume.shape[0]):
            for y_idx in range(volume.shape[1]):
                z_coords = np.where(volume[x_idx, y_idx, :] == 1)[0]
                if z_coords.size > 0:
                    height_map[x_idx, y_idx] = z_coords.max()
        if np.isnan(height_map).all():
            return np.zeros(volume.shape[:2], dtype=float)
        return height_map

    topdown_maps = [build_topdown_height(volume) for volume in list_3d]
    change_points = np.zeros(len(time_points), dtype=bool)
    for idx in range(1, len(label_grid)):
        change_points[idx] = not np.array_equal(lesion_sizes[idx], lesion_sizes[idx - 1])

    def raw_aspect_for(frame_idx):
        spacing = _get_voxel_spacing(heatmap_affines[frame_idx])
        return spacing[0] / spacing[1]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(heatmaps[0], cmap='copper', vmin=0, vmax=1, aspect=raw_aspect_for(0), animated=True)

    img_h_px, img_w_px = heatmaps[0].shape
    col_spacing0 = _get_voxel_spacing(heatmap_affines[0])[1]
    bar_length_mm = _nice_scale_length_mm(img_w_px * col_spacing0)
    bar_length_px0 = bar_length_mm / col_spacing0

    bar_x0 = img_w_px * 0.05
    bar_y = img_h_px * 0.92
    scale_line, = ax1.plot([bar_x0, bar_x0 + bar_length_px0], [bar_y, bar_y], color='white', lw=3, solid_capstyle='butt')
    scale_text = ax1.text(bar_x0 + bar_length_px0 / 2, bar_y - img_h_px * 0.03, f"{bar_length_mm:.0f} mm", color='white', ha='center', va='bottom', fontsize=10)

    ax1.set_title(f"Predicted Lesion Heatmap {patient_idx.item()} frame 0/{len(heatmaps)}")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im_label = ax2.imshow(label_grid[0], cmap='copper', vmin=0, vmax=1, aspect=label_aspect)
    ax2.set_title("Lesion Label of nnUNet")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    nx, ny, nz = list_3d[0].shape
    cube_extent = max(nx, ny, nz, 1)
    x_scale = cube_extent / max(nx, 1)
    y_scale = cube_extent / max(ny, 1)
    z_scale = cube_extent / max(nz, 1)
    xs, ys, zs = np.where(list_3d[0] == 1)
    scatter = ax3.scatter(xs * x_scale, ys * y_scale, zs * z_scale)
    ax3.set_title("Lesion Label in 3D space")
    ax3.set_xlim3d(0, cube_extent)
    ax3.set_ylim3d(0, cube_extent)
    ax3.set_zlim3d(0, cube_extent)
    ax3.set_box_aspect((1, 1, 1))

    x_plane = np.linspace(0, cube_extent, 10)
    y_plane = np.linspace(0, cube_extent, 10)
    X_p, Y_p = np.meshgrid(x_plane, y_plane)
    z_slice_idx0 = int(np.clip(((lesion_z_norm + 1) / 2) * (nz - 1), 0, nz - 1))
    Z_p = np.full_like(X_p, z_slice_idx0 * z_scale)
    ax3.plot_surface(X_p, Y_p, Z_p, alpha=0.3, color='lightblue', antialiased=False, label="slice_of_heatmap")

    ax4 = fig.add_subplot(gs[0, 2])
    heatmap0_resized = _resize_2d(heatmaps[0], label_grid[0].shape)
    im_label_4 = ax4.imshow(heatmap0_resized, cmap='copper', vmin=0, vmax=1, aspect=label_aspect, animated=True)
    im_seg_4 = ax4.imshow(label_grid[0], cmap='Reds', vmin=0, vmax=1, alpha=0.5, aspect=label_aspect)
    ax4.set_title("Lesion Label + Prediction Overlap")
    ax4.axis('off')

    max_vol = max(lesion_sizes.max(), 1e-6) if lesion_sizes.size > 0 else 1e-6
    mid_y = max_vol / 2

    ax_timeline = fig.add_subplot(gs[1, 2])
    ax_timeline.fill_between(time_points, 0, lesion_sizes, color='skyblue', alpha=0.25)
    ax_timeline.plot(time_points, lesion_sizes, color='steelblue', lw=1.5)
    ax_timeline.plot(time_points, np.full_like(time_points, 0.5), color='lightgray', lw=1)
    ax_timeline.scatter(time_points[change_points], np.full(change_points.sum(), 0.5), marker='|', color='red', s=120)
    progress_line, = ax_timeline.plot([time_points[0], time_points[0]], [0.25, 0.75], color='royalblue', lw=3)
    current_marker, = ax_timeline.plot([time_points[0]], [0.5], marker='o', color='royalblue', markersize=8)
    ax_timeline.set_xlim(time_points[0], time_points[-1])
    ax_timeline.set_ylim(-max_vol * 0.1, max_vol * 1.2)
    ax_timeline.set_yticks(np.linspace(0, max_vol, 4))
    ax_timeline.set_xlabel('Time')
    ax_timeline.set_ylabel('Lesion Size')
    ax_timeline.set_title('Lesion Growth Time Evolution Timeline')
    ax_timeline.set_xticks(np.linspace(time_points[0], time_points[-1], 5))

    ax_topdown = fig.add_subplot(gs[1, 1])
    topdown_max = max(1, nz - 1)
    topdown_im = ax_topdown.imshow(topdown_maps[0], cmap='terrain', vmin=0, vmax=topdown_max, aspect=label_aspect)
    ax_topdown.set_title('Top-down lesion height map')
    ax_topdown.axis('off')

    fig.tight_layout()

    def update(i):
        if i == len(heatmaps) - 1:
            im.set_array(np.ones_like(heatmaps[0]))
            im_label.set_array(np.ones_like(label_grid[0]))
            im_label_4.set_array(np.ones_like(heatmap0_resized))
            im_seg_4.set_array(np.ones_like(label_grid[0]))
            scatter._offsets3d = ([], [], [])
            ax1.set_title("--- End of Sequence ---")
            progress_line.set_data([time_points[0], time_points[-1]], [mid_y, mid_y])
            current_marker.set_data([time_points[-1]], [mid_y])
            topdown_im.set_array(topdown_maps[-1])
            topdown_im.set_clim(0, topdown_max)
        else:
            ax1.set_aspect(raw_aspect_for(i))
            im.set_array(heatmaps[i])
            im_label.set_array(label_grid[i])
            im_label_4.set_array(_resize_2d(heatmaps[i], label_grid[i].shape))
            im_seg_4.set_array(label_grid[i])

            col_spacing_i = _get_voxel_spacing(heatmap_affines[i])[1]
            bar_length_px_i = bar_length_mm / col_spacing_i
            scale_line.set_xdata([bar_x0, bar_x0 + bar_length_px_i])
            scale_text.set_position((bar_x0 + bar_length_px_i / 2, bar_y - img_h_px * 0.03))

            xs, ys, zs = np.where(list_3d[i] == 1)
            scatter._offsets3d = (xs * x_scale, ys * y_scale, zs * z_scale)
            ax1.set_title(f'Predicted Lesion Heatmap {patient_idx.item()} frame {i:03d}/{len(heatmaps)}')
            progress_line.set_data([time_points[0], time_points[i]], [mid_y, mid_y])
            current_marker.set_data([time_points[i]], [mid_y])
            topdown_im.set_array(topdown_maps[i])
            topdown_im.set_clim(0, topdown_max)
        return [im, im_label, im_label_4, im_seg_4, scatter, progress_line, current_marker, topdown_im, scale_line, scale_text, ax1.title]

    ani = FuncAnimation(fig, update, frames=len(heatmaps), interval=50)

    # because in older versions of magick, the convert command is used instead of magick, we try both
    try:
        plt.rcParams['animation.convert_path'] = 'convert'
        writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
        file_name = f"/tmp/{epoch:04d}_epoch_lesion_heatmap_patient_{trj.patient_id}_{patient_idx.item()}_lesion_{trj.label_id}.gif"
        ani.save(file_name, writer=writer, dpi=50)
    except Exception:
        plt.rcParams['animation.convert_path'] = 'magick'
        writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
        file_name = f"/tmp/{epoch:04d}_epoch_lesion_heatmap_patient_{trj.patient_id}_{patient_idx.item()}_lesion_{trj.label_id}.gif"
        ani.save(file_name, writer=writer, dpi=50)

    mlflow.log_artifact(file_name)
    plt.close(fig)

