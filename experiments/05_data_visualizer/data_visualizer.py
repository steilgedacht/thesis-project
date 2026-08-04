import sys
sys.path.insert(1, '/home/benjaminb/Dokumente/JKU/Semester_9/Practical_Work')
from utils.patient import Patient
from utils.mri_dataloader import MRI_Dataloader
import mlflow
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import numpy as np
from train import load_config
import tqdm
import gc
from scipy import ndimage


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

def plot_lesion_time_evolution(trj,  config):    
    patient = Patient(trj.patient_id)

    allowed_dates = trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates
    allowed_bool = [date in allowed_dates for date in patient.dates]

    full_matrix_labels = trj.load_labels_for_inr(absolute_day_number=True)

    # Shape of the label grid the trajectory was resampled to for the INR.
    label_shape = np.array(full_matrix_labels[0][0].shape)  # (X, Y, Z)

    # get the height (z index, shared by raw MRI and label grid) where the lesion is located
    center_of_mass = np.stack([
        ndimage.center_of_mass(full_matrix_labels[i][0])[-1]
        for i in range(len(full_matrix_labels))
    ])
    avg_center_of_mass = float(np.mean(np.sort(center_of_mass)[1:-1]))
    # normalize to [-1, 1] using the label grid's actual z-extent, not a fixed 50
    lesion_z_norm = avg_center_of_mass / max(label_shape[-1] - 1, 1) * 2 - 1

    # each raw MRI slice keeps its own affine, since spacing can vary per scan
    all_patient_samples = []
    for i in range(len(patient.samples)):
        if allowed_bool[i]:
            p, affine = patient.samples[i].load_mri(zoomed=True, affine=True)
            z_idx = int(np.clip(center_of_mass[sum(allowed_bool[:i])], 0, p.shape[-1] - 1))
            all_patient_samples.append((p[:, :, z_idx], affine))

    # reference spacing (raw MRI vs. label grid), used only to scale the 3D/topdown plots
    raw_shape = np.array(patient.samples[0].load_mri(zoomed=True).shape)
    raw_spacing_ref = _get_voxel_spacing(all_patient_samples[0][1])
    label_spacing = raw_spacing_ref * (raw_shape / label_shape)

    heatmaps = []
    heatmap_affines = []
    labels_list = []
    list_3d = []
    lesion_sizes = []

    time_points = np.linspace(0, full_matrix_labels[-1][1] + 180, config.time_evolution_steps)

    for t in time_points:
        for i, element in enumerate(reversed(full_matrix_labels), start=1):
            if element[1] <= t:
                z_size = element[0].shape[-1]
                z_idx = int(np.clip(((lesion_z_norm + 1) / 2) * (z_size - 1), 0, z_size - 1))
                labels_list.append(element[0][:, :, z_idx])
                list_3d.append(element[0])
                lesion_sizes.append(float(element[0].sum()))
                frame_img, frame_affine = all_patient_samples[len(all_patient_samples) - i]
                heatmaps.append(frame_img)
                heatmap_affines.append(frame_affine)
                break

    heatmaps = np.array(heatmaps)
    heatmap_max = np.max(heatmaps, axis=(1, 2))
    heatmap_max[heatmap_max == 0] = 1.0  # avoid divide-by-zero on empty/edge slices
    heatmaps = heatmaps / heatmap_max[:, np.newaxis, np.newaxis]
    label_grid = np.array(labels_list)
    list_3d = np.array(list_3d)
    lesion_sizes = np.array(lesion_sizes, dtype=float)  # voxel counts from the label grid
    voxel_volume_mm3 = label_spacing[0] * label_spacing[1] * label_spacing[2]
    lesion_sizes_cm3 = lesion_sizes * voxel_volume_mm3 / 1000.0  # mm^3 -> cm^3

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
        change_points[idx] = not np.array_equal(lesion_sizes_cm3[idx], lesion_sizes_cm3[idx - 1])

    def raw_aspect_for(frame_idx):
        sp = _get_voxel_spacing(heatmap_affines[frame_idx])
        return sp[0] / sp[1]

    label_aspect = label_spacing[1] / label_spacing[0]

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(heatmaps[0], cmap='copper', vmin=0, vmax=1, aspect=raw_aspect_for(0), animated=True)

    img_h_px, img_w_px = heatmaps[0].shape
    col_spacing0 = _get_voxel_spacing(heatmap_affines[0])[1]  # mm per pixel, horizontal axis
    bar_length_mm = _nice_scale_length_mm(img_w_px * col_spacing0)
    bar_length_px0 = bar_length_mm / col_spacing0

    bar_x0 = img_w_px * 0.05
    bar_y = img_h_px * 0.92
    scale_line, = ax1.plot([bar_x0, bar_x0 + bar_length_px0], [bar_y, bar_y],
                            color='white', lw=3, solid_capstyle='butt')
    scale_text = ax1.text(bar_x0 + bar_length_px0 / 2, bar_y - img_h_px * 0.03,
                        f"{bar_length_mm:.0f} mm", color='white', ha='center', va='bottom', fontsize=10)



    ax1.set_title(f"MRI Scan {trj.patient_id} frame 0/{len(heatmaps)}")
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    im_label = ax2.imshow(label_grid[0], cmap='copper', vmin=0, vmax=1, aspect=label_aspect)
    ax2.set_title("Lesion Label of nnUNet")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    nx, ny, nz = list_3d[0].shape
    xs, ys, zs = np.where(list_3d[0] == 1)
    scatter = ax3.scatter(xs, ys, zs=zs)
    ax3.set_title("Lesion Label in 3D space")
    ax3.set_xlim3d(0, nx)
    ax3.set_ylim3d(0, ny)
    ax3.set_zlim3d(0, nz)
    ax3.set_box_aspect((nx * label_spacing[0], ny * label_spacing[1], nz * label_spacing[2]))

    x_plane = np.linspace(0, nx, 10)
    y_plane = np.linspace(0, ny, 10)
    X_p, Y_p = np.meshgrid(x_plane, y_plane)
    z_slice_idx0 = int(np.clip(((lesion_z_norm + 1) / 2) * (nz - 1), 0, nz - 1))
    Z_p = np.full_like(X_p, z_slice_idx0)
    ax3.plot_surface(X_p, Y_p, Z_p, alpha=0.3, color='lightblue', antialiased=False, label="slice_of_heatmap")

    ax4 = fig.add_subplot(gs[0, 2])
    heatmap0_resized = _resize_2d(heatmaps[0], label_grid[0].shape)
    im_label_4 = ax4.imshow(heatmap0_resized, cmap='copper', vmin=0, vmax=1, aspect=label_aspect, animated=True)
    im_seg_4 = ax4.imshow(label_grid[0], cmap='Reds', vmin=0, vmax=1, alpha=0.5, aspect=label_aspect)
    ax4.set_title("Lesion Label + MRI Overlap")
    ax4.axis('off')

    max_vol = max(lesion_sizes_cm3.max(), 1e-6)
    mid_y = max_vol / 2  # height for the reference/progress line, scaled to real data now    

    ax_timeline = fig.add_subplot(gs[1, 2])
    ax_timeline.fill_between(time_points, 0, lesion_sizes_cm3, color='skyblue', alpha=0.25)
    ax_timeline.plot(time_points, lesion_sizes_cm3, color='steelblue', lw=1.5)
    ax_timeline.plot(time_points, np.full_like(time_points, 0.5), color='lightgray', lw=1)
    ax_timeline.scatter(time_points[change_points], np.full(change_points.sum(), 0.5), marker='|', color='red', s=120)
    progress_line, = ax_timeline.plot([time_points[0], time_points[0]], [0.25, 0.75], color='royalblue', lw=3)
    current_marker, = ax_timeline.plot([time_points[0]], [0.5], marker='o', color='royalblue', markersize=8)
    ax_timeline.set_xlim(time_points[0], time_points[-1])
    ax_timeline.set_ylim(-max_vol * 0.1, max_vol * 1.2)
    ax_timeline.set_yticks(np.linspace(0, max_vol, 4))
    ax_timeline.set_xlabel('Time')
    ax_timeline.set_ylabel('Lesion Size in cm³')
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
            scatter._offsets3d = (xs, ys, zs)
            ax1.set_title(f'Predicted Lesion Heatmap {trj.patient_id} frame {i:03d}/{len(heatmaps)}')
            progress_line.set_data([time_points[0], time_points[i]], [mid_y, mid_y])
            current_marker.set_data([time_points[i]], [mid_y])
            topdown_im.set_array(topdown_maps[i])
            topdown_im.set_clim(0, topdown_max)
        return [im, im_label, im_label_4, im_seg_4, scatter, progress_line, current_marker, topdown_im, scale_line, scale_text, ax1.title]
    ani = FuncAnimation(fig, update, frames=len(heatmaps), interval=50)

    try:
        plt.rcParams['animation.convert_path'] = 'convert'
        writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
        file_name = f"/tmp/lesion_heatmap_{sum(allowed_bool):02d}_lesions_patient_{trj.patient_id}_{trj.label_id}.gif"
        ani.save(file_name, writer=writer, dpi=50)
    except Exception:
        plt.rcParams['animation.convert_path'] = 'magick'
        writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
        file_name = f"/tmp/lesion_heatmap_{sum(allowed_bool):02d}_lesions_patient_{trj.patient_id}_{trj.label_id}.gif"
        ani.save(file_name, writer=writer, dpi=50)

    mlflow.log_artifact(file_name)
    plt.close(fig)

    return mlflow.get_artifact_uri(file_name)

mri_dataloader = MRI_Dataloader()
config = load_config("configs/03_data_visualizer/data_visualizer.py")

mlflow.set_experiment("data_visualizer")
mlflow.set_tracking_uri(config.mlflow_tracking_uri)


with mlflow.start_run(run_name="data_visualizer"):
    for i, trj in tqdm.tqdm(enumerate(mri_dataloader.iterate_trajectories()), total=len(mri_dataloader.lesion_trajectory_paths)):
        if len(trj.allowed_dates) < 4: continue
        with mlflow.start_span(f"Patient {trj.patient_id}, Lesion {trj.label_id}") as span:
            path = plot_lesion_time_evolution(trj, config=config)
            span.set_outputs({"chart_artifact": path})
        gc.collect()
