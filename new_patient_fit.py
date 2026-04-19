import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mri_dataloader import MRI_Dataloader
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from umap import UMAP

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lesion_INR_Training")


def visualize_embedding_change(model, patient_idx_tensor, train_loader):
    """Visualize embedding before/after training using UMAP projection"""
    
    # Get all embeddings from the model to fit UMAP
    all_embeddings = model.latent_vectors.weight.detach().cpu().numpy()
    
    # Fit UMAP on all embeddings
    umap = UMAP(n_components=2, random_state=42)
    all_embeddings_umap = umap.fit_transform(all_embeddings)
    
    # Get the specific patient's embedding (already fine-tuned)
    final_embedding = model.latent_vectors(patient_idx_tensor).detach().cpu().numpy()
    final_umap = umap.transform(final_embedding)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot all embeddings
    ax1.scatter(all_embeddings_umap[:, 0], all_embeddings_umap[:, 1], alpha=0.5, s=30, label='Other patients')
    ax1.scatter(final_umap[:, 0], final_umap[:, 1], color='red', s=200, marker='*', label='New patient (after)', edgecolors='black', linewidth=2)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_title('Embedding in UMAP Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot embedding dimensions heatmap
    fig2, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    # Show the full 128-d embedding as heatmap
    axes[0].imshow(final_embedding.reshape(1, -1), cmap='coolwarm', aspect='auto')
    axes[0].set_title('Fine-tuned Embedding (128 dimensions)')
    axes[0].set_ylabel('Patient')
    axes[0].set_xlabel('Embedding Dimension')
    
    # Show variance across all embeddings
    embedding_std = np.std(all_embeddings, axis=0)
    axes[1].plot(embedding_std)
    axes[1].set_title('Std Dev Across All Embeddings (shows which dims vary most)')
    axes[1].set_xlabel('Embedding Dimension')
    axes[1].set_ylabel('Std Dev')
    
    mlflow.log_figure(fig, "embedding_pca_projection.png")
    mlflow.log_figure(fig2, "embedding_heatmap.png")
    plt.close(fig)
    plt.close(fig2)


class LesionDataset(Dataset):
    def __init__(self, trajectories, device='cuda', context_radius=5, background_samples_proportion=1, mode='train', val_date_idx=None, val_end_date_idx=None):
        self.device = device
        self.trajectories = trajectories
        self.shape = (500, 500, 50)
        self.meshgrid = np.meshgrid(np.linspace(0, 1, self.shape[0]),
                              np.linspace(0, 1, self.shape[1]),
                              np.linspace(0, 1, self.shape[2]),
                              indexing='ij')
        self.context_radius = context_radius
        self.background_samples_proportion = background_samples_proportion
        self.patient_to_idx = {p.patient_id: i for i, p in enumerate(trajectories)}
        self.mode = mode

        if val_date_idx is None:
            self.val_date_idx = [np.random.choice(trj.dates[1:-1]) for trj in trajectories]
        else:
            self.val_date_idx = val_date_idx

        if val_end_date_idx is None:
            self.val_end_date_idx = np.random.choice(range(len(trajectories)), size=len(trajectories) // 10, replace=False)
        else:
            self.val_end_date_idx = val_end_date_idx
        
        if self.mode == 'valid_extrapolation':
            self.trajectories = [trj for i, trj in enumerate(trajectories) if i in self.val_end_date_idx]

    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trj = self.trajectories[idx]
        patient_idx = self.patient_to_idx[trj.patient_id]

        if self.mode == 'train':
            if idx in self.val_end_date_idx:
                random_time_point = str(np.random.choice(trj.dates[:-1]))            
            else:
                random_time_point = str(np.random.choice(trj.dates))
        else:
            if self.mode == 'valid_extrapolation':
                random_time_point = str(trj.dates[-1])
            if self.mode == 'valid_interpolation':
                random_time_point = str(self.val_date_idx[idx])

        labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point)
        
        lesion_mask = labels > 0.5        
        pos_coords = np.argwhere(lesion_mask)

        border_samples = binary_dilation(labels) - labels
        border_samples_coords = np.argwhere(border_samples)

        MAX_SAMPLES = 2500
        num_neg_needed = MAX_SAMPLES // 2
        neg_coords = border_samples_coords.tolist()

        if len(neg_coords) > num_neg_needed // 2:
            neg_coords = [neg_coords[i] for i in np.random.choice(len(neg_coords), size=num_neg_needed // 2, replace=True).tolist()]

        while len(neg_coords) < num_neg_needed:
            candidate_coords = np.array([
                np.random.randint(0, s, num_neg_needed) for s in self.shape
            ]).T
            is_bg = labels[candidate_coords[:,0], candidate_coords[:,1], candidate_coords[:,2]] <= 0.5
            neg_coords.extend(candidate_coords[is_bg])
        
        neg_coords = np.array(neg_coords)[:num_neg_needed]
        
        if len(pos_coords) == 0:
            all_sampled_indices = np.concatenate([neg_coords, neg_coords], axis=0)
        else:
            pos_idx = np.random.choice(len(pos_coords), size=MAX_SAMPLES//2, replace=True)
            pos_sampled = pos_coords[pos_idx]
            all_sampled_indices = np.vstack([pos_sampled, neg_coords])
        
        i, j, k = all_sampled_indices.T
        coords = np.stack([
            self.meshgrid[0][i, j, k] * 2 - 1,
            self.meshgrid[1][i, j, k] * 2 - 1,
            self.meshgrid[2][i, j, k] * 2 - 1,
            np.full(len(i), time_point)
        ], axis=1)
        
        labels_sampled = labels[i, j, k]

        return coords.astype(np.float32), labels_sampled.astype(np.float32), np.int64((patient_idx * 100) + self.trajectories[idx].label_id)

def plot_lesion_time_evolution(model, epoch, sample_idx, data_loader, steps=150):
    coords, labels, patient_idx = data_loader.dataset[sample_idx]
    coords = torch.from_numpy(coords).unsqueeze(0).float().to(device)
    labels = torch.from_numpy(labels).unsqueeze(0).float().unsqueeze(-1).to(device)
    patient_idx = torch.tensor(patient_idx).unsqueeze(0).to(device)
    
    full_matrix_labels = data_loader.dataset.trajectories[sample_idx].load_labels_for_inr()

    SIDE_LENGTH = 500

    heatmaps = []
    labels_list = []
    list_3d = []
    
    with torch.no_grad():

        # get the height where the lesion is located
        center_of_mass = torch.mean(coords.squeeze()[:len(labels.squeeze())//2,:3], axis=0).detach().cpu().numpy()
        lesion_z = center_of_mass[2]
        
        meshgrid = np.meshgrid(np.linspace(0, 1, SIDE_LENGTH), np.linspace(0, 1, SIDE_LENGTH), indexing='ij')
        x = meshgrid[0].flatten() * 2 - 1
        y = meshgrid[1].flatten() * 2 - 1
        z = np.full_like(x, lesion_z)

        for t in np.linspace(-1,1, steps):

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

            heatmap_preds = heatmap_preds.reshape(SIDE_LENGTH, SIDE_LENGTH)
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
    ax2.set_title("Ground Truth in 3D space")
    ax3.set_xlim3d(0, 500)
    ax3.set_ylim3d(0, 500)
    ax3.set_zlim3d(0, 50)

    x_plane = np.linspace(0, 500, 10)
    y_plane = np.linspace(0, 500, 10)
    X_p, Y_p = np.meshgrid(x_plane, y_plane)
    Z_p = np.full_like(X_p, int(((lesion_z + 1) / 2) * 50)) 
    ax3.plot_surface(X_p, Y_p, Z_p, alpha=0.3, color='lightblue', antialiased=False, label="slice_of_heatmap")

    ax4 = fig.add_subplot(2, 2, 4)
    im_label_4 = ax4.imshow(heatmaps[0].repeat(500//SIDE_LENGTH,axis=0).repeat(500//SIDE_LENGTH,axis=1), cmap='copper', vmin=0, vmax=1, animated=True)
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
            im_label_4.set_array(heatmaps[i].repeat(500//SIDE_LENGTH,axis=0).repeat(500//SIDE_LENGTH,axis=1))
            im_seg_4.set_array(label_grid[i])
            xs, ys, zs = np.where(list_3d[i]==1)
            scatter._offsets3d = (xs, ys, zs)

            ax1.set_title(f'Predicted Lesion Heatmap {patient_idx.item()} frame {i:03d}/{len(heatmaps)}')
        return [im, im_label, im_label_4, im_seg_4, ax1.title]
    
    ani = FuncAnimation(fig, update, frames=len(heatmaps), interval=50)

    plt.rcParams['animation.convert_path'] = 'magick'
    writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
    file_name = f"/tmp/{epoch:04d}_epoch_lesion_heatmap_patient_{patient_idx.item()}_sample_{sample_idx}.gif"
    ani.save(file_name, writer=writer, dpi=50)

    mlflow.log_artifact(file_name)
    
    plt.close(fig)

class Loss_BCE_Dice():
    def __init__(self, loss_fn_1=nn.BCEWithLogitsLoss()):
        self.loss_fn_1 = loss_fn_1
        self.loss_bce = 0
        self.loss_dice = 0
    
    def __call__(self, pred, target):
        self.loss_bce = self.loss_fn_1(pred, target)
        self.loss_dice = self.dice_loss(pred, target)
        return self.loss_bce + self.loss_dice

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))



def fit_trajectory_to_model(model, train_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.AdamW(model.latent_vectors.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.00001)    
    criterion = Loss_BCE_Dice()

    model.to(device)
    losses = []

    _, _, P_ID = next(iter(train_loader))
    P_ID = P_ID[0].item()

    initial_embedding = model.latent_vectors.weight[P_ID].clone().detach().cpu().numpy()
    embedding_history = [initial_embedding]


    # freeze all layers except the embedding layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.latent_vectors.parameters():
        param.requires_grad = True
    
    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": model.layers[0].linear.out_features,
        "omega_0": model.omega_0,
        "batch_size": train_loader.batch_size
    })
    
    global_step = 0

    monitoring_samples = np.random.choice(len(train_loader), size=1)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):

            coords = coords.to(device)
            labels = labels.to(device).unsqueeze(-1)
            patient_idx = patient_idx.to(device)

            if coords.shape[1] == 0:
                continue

            optimizer.zero_grad()

            predictions = model(coords, patient_idx)

            loss = criterion(predictions, labels)
            loss.backward()

            z_patient = model.latent_vectors(patient_idx[0:1])  # Get first sample's patient idx

            # Log embedding stats
            mlflow.log_metrics({
                "embedding_l2_norm": z_patient.norm(p=2).item(),
                "embedding_max": z_patient.max().item(),
                "embedding_min": z_patient.min().item(),
                "embedding_mean": z_patient.mean().item(),
                "embedding_std": z_patient.std().item(),
            }, step=global_step)

            # Log gradient statistics (before optimizer.step())
            for name, param in model.latent_vectors.named_parameters():
                if param.grad is not None:
                    mlflow.log_metrics({
                        f"grad_{name}_norm": param.grad.norm(p=2).item(),
                        f"grad_{name}_max": param.grad.max().item(),
                        f"grad_{name}_min": param.grad.min().item(),
                    }, step=global_step)
            
            optimizer.step()
            total_loss += loss.item()

            mlflow.log_metrics(
                {
                    "training_loss" : loss.item(),
                    "training_loss_bce" : criterion.loss_bce.item(),
                    "training_loss_dice" : criterion.loss_dice.item(),
                    "number_of_correctly_predicted_1_labels" : (predictions>0.5).sum().item() / (labels > 0.5).sum().item(),
                    "number_of_correctly_predicted_0_labels" : (predictions<=0.5).sum().item() / (labels <= 0.5).sum().item()
                },
                step=global_step
            )

            global_step += 1

            if i % 10 == 0:
                del coords, labels, patient_idx, predictions, loss
                torch.cuda.empty_cache()

            current_embedding = model.latent_vectors.weight[P_ID].detach().cpu().numpy()
            embedding_history.append(current_embedding)


        model.eval()
        torch.cuda.empty_cache()

        # if epoch % 20 == 0 and epoch != 0:
        #     with torch.no_grad():
        #         for m in monitoring_samples:
        #             plot_lesion_time_evolution(model, epoch, m, train_loader)

        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    with torch.no_grad():
        embedding_history = np.array(embedding_history)
        umap = UMAP(n_components=2, random_state=42)
        trajectory_umap = umap.fit_transform(embedding_history)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(trajectory_umap[:, 0], trajectory_umap[:, 1], 'o-', markersize=5)
        ax.scatter(trajectory_umap[0, 0], trajectory_umap[0, 1], color='green', s=200, marker='o', label='Start')
        ax.scatter(trajectory_umap[-1, 0], trajectory_umap[-1, 1], color='red', s=200, marker='*', label='End')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Embedding Trajectory During Fine-tuning')
        ax.legend()
        ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "embedding_trajectory.png")
        plt.close(fig)
        visualize_embedding_change(model, torch.tensor([P_ID]).to(device), train_loader)
    
    mlflow.pytorch.log_model(model, name="lesion_inr_model")
    
    for m in monitoring_samples:
        plot_lesion_time_evolution(model, epoch, m, train_loader)

    return losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with mlflow.start_run(tags={"type" : "new_patient_fit"}):
    mri_dataloader = MRI_Dataloader()
    trj = [mri_dataloader.find_lesion_trajectory('YG_WCONBI42RLEZ', 1)] * 10
    
    train_dataset = LesionDataset(trj, context_radius=5, background_samples_proportion=1, device=device, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=7, prefetch_factor=2, pin_memory=True, persistent_workers=True)

    run_id = "8fe747bb5b064f70be27b66dcb418617"
    model_uri = f"runs:/{run_id}/lesion_inr_model"
    model = mlflow.pytorch.load_model(model_uri)

    losses = fit_trajectory_to_model(model, train_loader,  epochs=75, lr=5e-3, device=device)


