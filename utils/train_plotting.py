import torch
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter

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

def plot_heatmap(model, labels, coords, patient_idx, epoch, sample_id, time_point=None):
    # get the height where the lesion is located
    center_of_mass = torch.mean(coords.squeeze()[:len(labels.squeeze())//2,:3], axis=0).detach().cpu().numpy()
    lesion_z = center_of_mass[2]

    if time_point is not None:
        sample_id = str(sample_id) + "_time_point_" + str(time_point)
    
    meshgrid = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500), indexing='ij')
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

    heatmap_preds = heatmap_preds.reshape(500, 500)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(heatmap_preds, cmap='copper', vmin=0, vmax=1)
    ax.set_title(f'Predicted Lesion Heatmap {patient_idx.item()}')
    ax.axis('off')
    mlflow.log_figure(fig, f"{epoch:04d}_epoch_lesion_heatmap_patient_{patient_idx.item()}_sample_{sample_id}.png")
    plt.close(fig)
    
def visualize_samples(model, epoch, monitoring_samples, data_loader, text):
    for sample_idx in monitoring_samples:
        coords, labels, patient_idx = data_loader.dataset[sample_idx]
        coords = torch.from_numpy(coords).float().unsqueeze(0).to(device)
        labels = torch.from_numpy(labels).float().unsqueeze(0).unsqueeze(-1).to(device)
        patient_idx = torch.tensor(patient_idx).unsqueeze(0).to(device)
        predictions = model(coords, patient_idx)
        plot_predictions(predictions, labels, coords, epoch, f"{text}_patient_{patient_idx.item()}_sample_{sample_idx}")


def plot_lesion_time_evolution(model, epoch, sample_idx, data_loader, steps=100, side_length=50):
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

    ax1 = fig.add_subplot(2, 2, 1)
    im = ax1.imshow(heatmaps[0], cmap='copper', vmin=0, vmax=1, animated=True)
    ax1.set_title(f"Predicted Lesion Heatmap {patient_idx.item()} frame 0/{len(heatmaps)}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 2, 2)
    im_label = ax2.imshow(label_grid[0], cmap='copper', vmin=0, vmax=1)
    ax2.set_title("Ground Truth")
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    xs, ys, zs = np.where(list_3d[0]==1)
    scatter = ax3.scatter(xs, ys, zs=zs)
    ax3.set_title("Ground Truth in 3D space")
    ax3.set_xlim3d(0, 500)
    ax3.set_ylim3d(0, 500)
    ax3.set_zlim3d(0, 50)

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

