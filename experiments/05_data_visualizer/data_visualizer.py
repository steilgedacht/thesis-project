from mri_dataloader import MRI_Dataloader
import mlflow
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_lesion_time_evolution(sample_idx, data_loader, steps=100, side_length=50):
    coords, labels, patient_idx = data_loader.dataset[sample_idx]
    coords = torch.from_numpy(coords).unsqueeze(0).float().to(device)
    labels = torch.from_numpy(labels).unsqueeze(0).float().unsqueeze(-1).to(device)
    patient_idx = torch.tensor(patient_idx).unsqueeze(0).to(device)
    
    full_matrix_labels = data_loader.dataset.trajectories[sample_idx].load_labels_for_inr(absolute_day_number=True)

    heatmaps = []
    labels_list = []
    list_3d = []
    
    with torch.no_grad():

        # get the height where the lesion is located
        center_of_mass = torch.mean(coords.squeeze()[:len(labels.squeeze())//2,:3], axis=0).detach().cpu().numpy()
        lesion_z = center_of_mass[2]
        
        meshgrid = np.meshgrid(np.linspace(0, 1, side_length), np.linspace(0, 1, side_length), indexing='ij')
        x = meshgrid[0].flatten() * 2 - 1
        y = meshgrid[1].flatten() * 2 - 1
        z = np.full_like(x, lesion_z)

        for t in np.linspace(0, 3650, steps):

            time_point = np.full_like(x, t) 

            hm_coords = np.stack([x, y, z, time_point], axis=1)
            coords_tensor = torch.from_numpy(hm_coords).float().to(next(model.parameters()).device).unsqueeze(0)

            with torch.no_grad():
                for slice_idx in range(0, coords_tensor.shape[1], 150_000):
                    slice_coords = coords_tensor[:, slice_idx:slice_idx+150_000]
                    slice_preds = model(slice_coords, patient_idx)
                    if slice_idx == 0:
                        heatmap_preds = slice_preds.cpu().numpy()
                    else:
                        heatmap_preds = np.concatenate([heatmap_preds, slice_preds.cpu().numpy()], axis=1)

            heatmap_preds = heatmap_preds.reshape(side_length, side_length)
            heatmaps.append(heatmap_preds)


            for element in reversed(full_matrix_labels):
                if element[1] <= t:
                    labels_list.append(element[0][:,:,int(((lesion_z + 1) / 2) * 50)])
                    list_3d.append(element[0])
                    break

    heatmaps = np.array(heatmaps)
    label_grid = np.array(labels_list)
    list_3d = np.array(list_3d)

    fig = plt.figure(figsize=(12, 12))


    ax3 = fig.add_subplot(2, 2, 1, projection='3d')
    xs, ys, zs = np.where(list_3d[0]==1)
    scatter = ax3.scatter(xs, ys, zs=zs)
    ax3.set_title("Ground Truth in 3D space")
    ax3.set_xlim3d(0, 500)
    ax3.set_ylim3d(0, 500)
    ax3.set_zlim3d(0, 50)
    
    ax2 = fig.add_subplot(2, 2, 2)
    im_label = ax2.imshow(label_grid[0], cmap='copper', vmin=0, vmax=1)
    ax2.set_title("Ground Truth")
    ax2.axis('off')

    ax1 = fig.add_subplot(2, 2, 3)
    im = ax1.imshow(heatmaps[0], cmap='copper', vmin=0, vmax=1, animated=True)
    ax1.set_title(f"Predicted Lesion Heatmap {patient_idx.item()} frame 0/{len(heatmaps)}")
    ax1.axis('off')

    x_plane = np.linspace(0, 500, 10)
    y_plane = np.linspace(0, 500, 10)
    X_p, Y_p = np.meshgrid(x_plane, y_plane)
    Z_p = np.full_like(X_p, int(((lesion_z + 1) / 2) * 50)) 
    ax3.plot_surface(X_p, Y_p, Z_p, alpha=0.3, color='lightblue', antialiased=False, label="slice_of_heatmap")

    ax4 = fig.add_subplot(2, 2, 4)
    im_label_4 = ax4.imshow(heatmaps[0].repeat(500//side_length,axis=0).repeat(500//side_length,axis=1), cmap='copper', vmin=0, vmax=1, animated=True)
    im_seg_4 = ax4.imshow(label_grid[0], cmap='Reds', vmin=0, vmax=1, alpha=0.5)
    ax4.set_title("Ground Truth")
    ax4.axis('off')


    fig.legend()

    def update(i):
        if i == len(heatmaps) - 1:
            im.set_array(np.ones_like(heatmaps[0]))
            im_label.set_array(np.ones_like(label_grid[0]))
            im_label_4.set_array(np.ones_like(heatmaps[0]))
            im_seg_4.set_array(np.ones_like(label_grid[0]))
            
            scatter._offsets3d = ([], [], [])
            
            ax1.set_title("--- End of Sequence ---")
        else:
            im.set_array(heatmaps[i])
            im_label.set_array(label_grid[i])
            im_label_4.set_array(heatmaps[i].repeat(500//side_length,axis=0).repeat(500//side_length,axis=1))
            im_seg_4.set_array(label_grid[i])
            xs, ys, zs = np.where(list_3d[i]==1)
            scatter._offsets3d = (xs, ys, zs)

            ax1.set_title(f'Predicted Lesion Heatmap {patient_idx.item()} frame {i:03d}/{len(heatmaps)}')
        return [im, im_label, im_label_4, im_seg_4, ax1.title]
    
    ani = FuncAnimation(fig, update, frames=len(heatmaps), interval=50)

    plt.rcParams['animation.convert_path'] = 'convert'
    writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
    file_name = f"/tmp/{epoch:04d}_epoch_lesion_heatmap_patient_{patient_idx.item()}_sample_{sample_idx}.gif"
    ani.save(file_name, writer=writer, dpi=50)

    mlflow.log_artifact(file_name)
    
    plt.close(fig)



mri_dataloader = MRI_Dataloader()

mri_dataloader.cache_lesion_trajectories_from_n_scans(n_scans=25, only_growing=True)
trajectories = mri_dataloader.cache_lesion_trajectories

mlflow.set_experiment("data_visualizer")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

for trajectory in trajectories:
    plot_lesion_time_evolution(sample_idx=trajectory.sample_idx, data_loader=mri_dataloader, steps=100, side_length=50)