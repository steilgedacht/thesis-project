import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from mri_dataloader import MRI_Dataloader, Patient
from scipy.ndimage import zoom
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import concurrent.futures
import threading


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lesion_INR_Training")

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.conditioning_lin = nn.Linear(latent_dim, out_features) 
        
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x, latent):
        modulation = self.conditioning_lin(latent).unsqueeze(1)
        return torch.sin(self.omega_0 * (self.linear(x) + modulation))

class LesionINR(nn.Module):
    def __init__(self, numpatients, latent_dim=128, input_dim=4, hidden_dim=512, output_dim=1, omega_0=30.0):
        super().__init__()
        self.latent_vectors = nn.Embedding(numpatients, latent_dim)
        self.first_layer = SirenLayer(input_dim, hidden_dim, latent_dim, is_first=True, omega_0=omega_0)
        self.layers = nn.ModuleList([
            SirenLayer(hidden_dim, hidden_dim, latent_dim, is_first=False, omega_0=omega_0)
            for _ in range(4)
        ])
        
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.omega_0 = omega_0

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        torch.nn.init.normal_(self.latent_vectors.weight, std=1.0 / np.sqrt(latent_dim))

    def forward(self, x, patient_idx):
        z = self.latent_vectors(patient_idx)
        x = self.first_layer(x, z)
        for layer in self.layers:
            x = layer(x, z)
            
        return self.final_layer(x)

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
    
    def __getitem__(self, idx, full_data=False):
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

        try:
            labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point)
        except:
            patient = Patient(trj.patient_id)
            patient.merge_lesion_to_trajectory()
            try:
                labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point)
            except:
                print(trj.patient_id, random_time_point)
                raise 

        # interpoltate the labels to the shape
        original_shape = labels.shape
        zoom_factors = (
            self.shape[0] / original_shape[0],
            self.shape[1] / original_shape[1],
            self.shape[2] / original_shape[2]
        )
        
        labels = zoom(labels, zoom_factors, order=1)
        x, y, z = np.copy(self.meshgrid)
        

        # 1. Labels binarisieren für schnellere Logik
        lesion_mask = labels > 0.5
        
        # 2. Koordinaten der positiven Voxel (Läsion)
        pos_coords = np.argwhere(lesion_mask)
        
        # 3. Negative Voxel (Hintergrund) finden
        # Trick: Statt argwhere auf dem ganzen Bild, sample einfach zufällige Punkte
        # und schaue, ob sie NICHT in der Maske liegen.
        MAX_SAMPLES = 1000
        num_neg_needed = MAX_SAMPLES // 2
        neg_coords = []
        while len(neg_coords) < num_neg_needed:
            candidate_coords = np.array([
                np.random.randint(0, s, num_neg_needed) for s in self.shape
            ]).T
            # Prüfe welche Kandidaten Hintergrund sind
            is_bg = labels[candidate_coords[:,0], candidate_coords[:,1], candidate_coords[:,2]] <= 0.5
            neg_coords.extend(candidate_coords[is_bg])
        
        neg_coords = np.array(neg_coords)[:num_neg_needed]
        
        # 4. Positive samples (mit Replacement falls Läsion klein ist)
        if len(pos_coords) == 0:
            all_sampled_indices = np.concatenate([neg_coords, neg_coords], axis=0)
        else:
            pos_idx = np.random.choice(len(pos_coords), size=MAX_SAMPLES//2, replace=True)
            pos_sampled = pos_coords[pos_idx]
            
            # 5. Zusammenführen
            all_sampled_indices = np.vstack([pos_sampled, neg_coords])
        
        # 6. Vektorisierte Extraktion aus meshgrid (Kein Loop!)
        i, j, k = all_sampled_indices.T
        coords = np.stack([
            self.meshgrid[0][i, j, k] * 2 - 1,
            self.meshgrid[1][i, j, k] * 2 - 1,
            self.meshgrid[2][i, j, k] * 2 - 1,
            np.full(len(i), time_point)
        ], axis=1)
        
        labels_sampled = labels[i, j, k]

        return coords.astype(np.float32), labels_sampled.astype(np.float32), patient_idx

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
    
def visualize_samples(model, epoch, monitoring_samples, dataloader, text):
    for sample_idx in monitoring_samples:
        coords, labels, patient_idx = dataloader.dataset[sample_idx]
        coords = torch.from_numpy(coords).float().unsqueeze(0).to(device)
        labels = torch.from_numpy(labels).float().unsqueeze(0).unsqueeze(-1).to(device)
        patient_idx = torch.tensor(patient_idx).unsqueeze(0).to(device)
        predictions = model(coords, patient_idx)
        plot_predictions(predictions, labels, coords, epoch, f"{text}_patient_{patient_idx.item()}_sample_{sample_idx}")
        plot_heatmap(model, labels, coords, patient_idx, epoch, sample_idx)

def validate_polation(dataloader, text, global_step, criterion):
    loss = 0
    for v_coords, v_labels, v_p_idx in tqdm(dataloader, desc=text, total=len(dataloader)):
        v_coords, v_labels, v_p_idx = v_coords.to(device), v_labels.unsqueeze(-1).to(device), v_p_idx.to(device)
        v_preds = model(v_coords, v_p_idx)
        loss += criterion.dice_loss(v_preds, v_labels).item()
    mlflow.log_metric(text.lower().replace(" ", "_"), loss / len(dataloader), step=global_step)

def plot_lesion_time_evolution(model, epoch, sample_idx, data_loader, steps=50):
    coords, labels, patient_idx = data_loader.dataset[sample_idx]
    coords = torch.from_numpy(coords).unsqueeze(0).float().to(device)
    labels = torch.from_numpy(labels).unsqueeze(0).float().unsqueeze(-1).to(device)
    patient_idx = torch.tensor(patient_idx).unsqueeze(0).to(device)
    
    full_matrix_labels = data_loader.dataset.trajectories[sample_idx].load_labels_for_inr()
    zoomed_full_matrix_labels = []
    for matrix in full_matrix_labels:
        original_shape = matrix[0].shape
        zoom_factors = (
            500 / original_shape[0],
            500 / original_shape[1],
            50 / original_shape[2]
        )
        zoomed_full_matrix_labels.append((zoom(matrix[0], zoom_factors, order=1), matrix[1]))

    heatmaps = []
    labels_list = []
    list_3d = []
    with torch.no_grad():

        # get the height where the lesion is located
        center_of_mass = torch.mean(coords.squeeze()[:len(labels.squeeze())//2,:3], axis=0).detach().cpu().numpy()
        lesion_z = center_of_mass[2]
        
        meshgrid = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 1, 500), indexing='ij')
        x = meshgrid[0].flatten() * 2 - 1
        y = meshgrid[1].flatten() * 2 - 1
        z = np.full_like(x, lesion_z)

        for t in np.linspace(-1,1, steps):

            time_point = np.full_like(x, t) 

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
            heatmaps.append(heatmap_preds)


            for element in reversed(zoomed_full_matrix_labels):
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
    im_label_4 = ax4.imshow(heatmaps[0], cmap='copper', vmin=0, vmax=1, animated=True)
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
            im_label_4.set_array(heatmaps[i])
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



def train_inr(model, train_loader, valid_interpolation_loader, valid_extrapolation_loader, epochs=100, lr=1e-3, device='cuda'):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)    
    criterion = Loss_BCE_Dice()

    model.to(device)
    losses = []
    
    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": model.layers[0].linear.out_features,
        "omega_0": model.omega_0,
        "batch_size": train_loader.batch_size
    })
    
    global_step = 0

    monitoring_samples = np.random.choice(len(train_loader), size=5, replace=False)

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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
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

        model.eval()
        torch.cuda.empty_cache()

        # semaphore = threading.Semaphore(2)

        with torch.no_grad():
            # visualize_samples(model, epoch, monitoring_samples, train_loader, "train")
            # visualize_samples(model, epoch, monitoring_samples, valid_interpolation_loader, "valid")
            # validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion)
            # validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion)
            plot_lesion_time_evolution(model, epoch, monitoring_samples[0], train_loader)

            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     def submit_with_semaphore(task, *args, **kwargs):
            #         with semaphore:
            #             return task(*args, **kwargs)
            #     futures = [
            #         executor.submit(submit_with_semaphore, visualize_samples, model, epoch, monitoring_samples, train_loader, "train"),
            #         executor.submit(submit_with_semaphore, visualize_samples, model, epoch, monitoring_samples, valid_interpolation_loader, "valid"),
            #         executor.submit(submit_with_semaphore, validate_polation, valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion),
            #         executor.submit(submit_with_semaphore, validate_polation, valid_interpolation_loader, "Valid Interpolation", global_step, criterion),
            #         executor.submit(submit_with_semaphore, plot_lesion_time_evolution, model, epoch, monitoring_samples[0], valid_interpolation_loader)
            #     ]
            #     for future in concurrent.futures.as_completed(futures):
            #         try:
            #             future.result()
            #             torch.cuda.empty_cache()
            #         except Exception as e:
            #             print(f"An error occurred: {e}")


        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=global_step)
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    mlflow.pytorch.log_model(model, name="lesion_inr_model")
    return losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with mlflow.start_run():
    mri_dataloader = MRI_Dataloader()
    mri_dataloader.cache_lesion_trajectories_from_n_scans(10)
    trajectories = mri_dataloader.cache_lesion_trajectories

    batchsize = 16
    epochs = 100
    
    train_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=6)
    
    valid_interpolation_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='valid_interpolation', val_date_idx=train_dataset.val_date_idx, val_end_date_idx=train_dataset.val_end_date_idx)
    valid_interpolation_loader = DataLoader(valid_interpolation_dataset, batch_size=batchsize, shuffle=False)

    valid_extrapolation_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='valid_extrapolation', val_date_idx=train_dataset.val_date_idx, val_end_date_idx=train_dataset.val_end_date_idx)
    valid_extrapolation_loader = DataLoader(valid_extrapolation_dataset, batch_size=batchsize, shuffle=False)

    model = LesionINR(len(trajectories), latent_dim=64, input_dim=4, hidden_dim=2048, output_dim=1)
    mlflow.log_params({
        "dataset_size": len(train_dataset), 
        "batch_size": batchsize,
        "latent_dim": model.latent_dim,
        "hidden_dim": model.hidden_dim,
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "omega" : model.omega_0,
        "device": device,
        "epochs": epochs
    })

    losses = train_inr(model, train_loader, valid_interpolation_loader, valid_extrapolation_loader, epochs=epochs, lr=1e-5, device=device)
    
    final_loss = losses[-1]
    mlflow.log_metric("final_train_loss", final_loss)
