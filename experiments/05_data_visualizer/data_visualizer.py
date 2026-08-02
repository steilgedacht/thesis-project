import sys
sys.path.insert(1, '/home/benjaminb/Dokumente/JKU/Semester_9/Practical_Work')
from utils.patient import Patient
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader import LesionDataset
from torch.utils.data import DataLoader
import mlflow
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import numpy as np
from train import load_config
import tqdm
import gc
from scipy import ndimage

def plot_lesion_time_evolution(patient_idx, data_loader, config, final_side_length=False):
    side_length = config.time_evolution_side_length if not final_side_length else config.full_size_side_length
    
    patient = Patient(data_loader.dataset.idx_to_patient[patient_idx.item() // 100])

    trj = data_loader.dataset.trajectories[patient_idx.item() // 100]
    allowed_dates = trj.allowed_dates if hasattr(trj, 'allowed_dates') else trj.dates
    allowed_bool = [date in allowed_dates for date in patient.dates]

    full_matrix_labels = trj.load_labels_for_inr(absolute_day_number=True)

    # get the height where the lesion is located
    center_of_mass = np.stack([ndimage.center_of_mass(full_matrix_labels[i][0])[-1] for i in range(len(full_matrix_labels)) ])
    avg_center_of_mass = int(np.mean(np.sort(center_of_mass)[1:-1]))
    lesion_z = avg_center_of_mass / 50 * 2 - 1
    

    all_patient_samples = [patient.samples[i].load_mri(zoomed=True)[:,:,int(center_of_mass[sum(allowed_bool[:i])])] for i in range(len(patient.samples)) if allowed_bool[i]]

    heatmaps = []
    labels_list = []
    list_3d = []
    lesion_sizes = []
    
    time_points = np.linspace(0, full_matrix_labels[-1][1] + 365, config.time_evolution_steps)

    for t in time_points:
        for i, element in enumerate(reversed(full_matrix_labels), start=1):
            if element[1] <= t:
                labels_list.append(element[0][:,:,int(((lesion_z + 1) / 2) * 50)])
                list_3d.append(element[0])
                lesion_sizes.append(float(element[0].sum()))
                heatmaps.append(all_patient_samples[len(all_patient_samples)-i])
                break

    heatmaps = np.array(heatmaps)
    heatmaps = heatmaps/np.max(heatmaps, axis=(1,2))[:,np.newaxis,np.newaxis]
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

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(heatmaps[0], cmap='copper', vmin=0, vmax=1, animated=True)
    ax1.set_title(f"Predicted Lesion Heatmap {patient_idx.item()} frame 0/{len(heatmaps)}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im_label = ax2.imshow(label_grid[0], cmap='copper', vmin=0, vmax=1)
    ax2.set_title("Ground Truth")
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    xs, ys, zs = np.where(list_3d[0] == 1)
    scatter = ax3.scatter(xs, ys, zs=zs)
    ax3.set_title("Ground Truth in 3D space")
    ax3.set_xlim3d(0, config.full_size_side_length)
    ax3.set_ylim3d(0, config.full_size_side_length)
    ax3.set_zlim3d(0, 50)

    x_plane = np.linspace(0, config.full_size_side_length, 10)
    y_plane = np.linspace(0, config.full_size_side_length, 10)
    X_p, Y_p = np.meshgrid(x_plane, y_plane)
    Z_p = np.full_like(X_p, int(((lesion_z + 1) / 2) * 50)) 
    ax3.plot_surface(X_p, Y_p, Z_p, alpha=0.3, color='lightblue', antialiased=False, label="slice_of_heatmap")

    ax4 = fig.add_subplot(gs[1, 1])
    im_label_4 = ax4.imshow(heatmaps[0].repeat(config.full_size_side_length // side_length, axis=0).repeat(config.full_size_side_length // side_length, axis=1), cmap='copper', vmin=0, vmax=1, animated=True)
    im_seg_4 = ax4.imshow(label_grid[0], cmap='Reds', vmin=0, vmax=1, alpha=0.5)
    ax4.set_title("Ground Truth")
    ax4.axis('off')

    ax_timeline = fig.add_subplot(gs[0, 2])
    ax_timeline.fill_between(time_points, 0, lesion_sizes, color='skyblue', alpha=0.25)
    ax_timeline.plot(time_points, lesion_sizes, color='steelblue', lw=1.5)
    ax_timeline.plot(time_points, np.full_like(time_points, 0.5), color='lightgray', lw=1)
    ax_timeline.scatter(time_points[change_points], np.full(change_points.sum(), 0.5), marker='|', color='red', s=120)
    progress_line, = ax_timeline.plot([time_points[0], time_points[0]], [0.25, 0.75], color='royalblue', lw=3)
    current_marker, = ax_timeline.plot([time_points[0]], [0.5], marker='o', color='royalblue', markersize=8)
    ax_timeline.set_xlim(time_points[0], time_points[-1])
    ax_timeline.set_ylim(-1, 2)
    ax_timeline.set_yticks([0, 0.5, 1])
    ax_timeline.set_xlabel('Time')
    ax_timeline.set_ylabel('Lesion Size')
    ax_timeline.set_title('Time Evolution Timeline')
    ax_timeline.set_xticks(np.linspace(time_points[0], time_points[-1], 5))

    ax_topdown = fig.add_subplot(gs[1, 2])
    topdown_max = max(1, list_3d[0].shape[2] - 1)
    topdown_im = ax_topdown.imshow(topdown_maps[0], cmap='terrain', vmin=0, vmax=topdown_max)
    ax_topdown.set_title('Top-down lesion height')
    ax_topdown.axis('off')

    fig.tight_layout()

    def update(i):
        if i == len(heatmaps) - 1:
            im.set_array(np.ones_like(heatmaps[0]))
            im_label.set_array(np.ones_like(label_grid[0]))
            im_label_4.set_array(np.ones_like(heatmaps[0]))
            im_seg_4.set_array(np.ones_like(label_grid[0]))
            scatter._offsets3d = ([], [], [])
            ax1.set_title("--- End of Sequence ---")
            progress_line.set_data([time_points[0], time_points[-1]], [0.5, 0.5])
            current_marker.set_data([time_points[-1]], [0.5])
            topdown_im.set_array(topdown_maps[-1])
            topdown_im.set_clim(0, topdown_max)
        else:
            im.set_array(heatmaps[i])
            im_label.set_array(label_grid[i])
            im_label_4.set_array(heatmaps[i].repeat(config.full_size_side_length // side_length, axis=0).repeat(config.full_size_side_length // side_length, axis=1))
            im_seg_4.set_array(label_grid[i])
            xs, ys, zs = np.where(list_3d[i] == 1)
            scatter._offsets3d = (xs, ys, zs)
            ax1.set_title(f'Predicted Lesion Heatmap {patient_idx.item()} frame {i:03d}/{len(heatmaps)}')
            progress_line.set_data([time_points[0], time_points[i]], [0.5, 0.5])
            current_marker.set_data([time_points[i]], [0.5])
            topdown_im.set_array(topdown_maps[i])
            topdown_im.set_clim(0, topdown_max)
        return [im, im_label, im_label_4, im_seg_4, scatter, progress_line, current_marker, topdown_im, ax1.title]
    
    ani = FuncAnimation(fig, update, frames=len(heatmaps), interval=50)

    # because in older versions of magick, the convert command is used instead of magick, we try both
    try:
        plt.rcParams['animation.convert_path'] = 'convert'
        writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
        file_name = f"/tmp/lesion_heatmap_{sum(allowed_bool):02d}_lesions_patient_{patient_idx.item():05d}_{data_loader.dataset.idx_to_patient[patient_idx.item() // 100]}.gif"
        ani.save(file_name, writer=writer, dpi=50)
    except:
        plt.rcParams['animation.convert_path'] = 'magick'
        writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
        file_name = f"/tmp/lesion_heatmap_{sum(allowed_bool):02d}_lesions_patient_{patient_idx.item():05d}_{data_loader.dataset.idx_to_patient[patient_idx.item() // 100]}.gif"
        ani.save(file_name, writer=writer, dpi=50)

    mlflow.log_artifact(file_name)
    
    plt.close(fig)


mri_dataloader = MRI_Dataloader()
mri_dataloader.cache_lesion_trajectories_from_n_scans(n_scans=4, only_growing=True)
trajectories = mri_dataloader.cache_lesion_trajectories

config = load_config("configs/03_data_visualizer/data_visualizer.py")

mlflow.set_experiment("data_visualizer")
mlflow.set_tracking_uri(config.mlflow_tracking_uri)

train_dataset = LesionDataset(
    trajectories, 
    dialation_iterations=config.dialation_iterations, 
    device=config.device, 
    mode='train', 
    background_samples=config.background_samples
)
train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batchsize, 
    shuffle=False, 
    num_workers=config.num_workers, 
    prefetch_factor=config.prefetch_factor, 
    pin_memory=config.pin_memory, 
    persistent_workers=config.persistent_workers
)

with mlflow.start_run(run_name="data_visualizer"):
    for i, (_, _, idx) in tqdm.tqdm(enumerate(train_loader), total=len(trajectories)):
        with mlflow.start_span(f"Patient {train_loader.dataset.idx_to_patient[idx.item() // 100]}, Lesion {idx.item() % 100}"):
            plot_lesion_time_evolution(idx, data_loader=train_loader, config=config, final_side_length=True)
        gc.collect()