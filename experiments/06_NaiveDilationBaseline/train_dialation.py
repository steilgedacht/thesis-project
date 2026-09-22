import sys
sys.path.insert(1, '/home/benjaminb/Dokumente/JKU/Semester_9/Practical_Work')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch
from utils.loss_bce_dice import Loss_BCE_Dice
from utils.mri_dataloader import MRI_Dataloader
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lesion_Dialation")


class NaiveDilationBaseline(nn.Module):
    def __init__(self, trajectories, shape=(500, 500, 50), pixels_per_day=0.05):
        super().__init__()
        self.shape = shape
        self.pixels_per_day = pixels_per_day  # Hyperparameter: Wie schnell wächst die Läsion pro Tag?
        
        # Wir speichern die allererste Maske und den Startzeitpunkt für jeden Patienten
        self.base_masks = {}
        self.base_times = {}
        
        print("Pre-loading base scans for naive baseline...")
        for idx, trj in enumerate(trajectories):
            # Nutze den eindeutigen ID-Key analog zu deinem Dataset
            label_key = (idx * 100) + trj.label_id
            
            # Lade den absolut ersten verfügbaren Scan des Patienten
            first_date = str(trj.allowed_dates[0])
            labels, time_point = trj.load_labels_for_inr(selected_date=first_date, absolute_day_number=True)
            
            self.base_masks[label_key] = torch.tensor(labels, dtype=torch.float32)
            self.base_times[label_key] = time_point

    def forward(self, x, patient_idx):
        """
        x: Shape [Batch, Num_Samples, 4] -> (x, y, z, t)
        patient_idx: Shape [Batch]
        """
        device = x.device
        batch_size, num_samples, _ = x.shape
        out = torch.zeros(batch_size, num_samples, 1, device=device)
        
        # Da wir im Grid-Raum [-1, 1] arbeiten, müssen wir die Koordinaten 
        # zurück in Pixel-Indizes (0 bis shape) rechnen, um die Dilation abzufragen.
        coords_xyz = x[..., :3] # [Batch, Num_Samples, 3]
        
        # Denormierung von [-1, 1] zu [0, shape-1]
        shape_tensor = torch.tensor(self.shape, device=device).view(1, 1, 3)
        pixel_coords = ((coords_xyz + 1.0) / 2.0) * (shape_tensor - 1.0)
        pixel_coords = torch.round(pixel_coords).long()
        
        # Begrenzen, um IndexErrors zu vermeiden
        for d in range(3):
            pixel_coords[..., d] = torch.clamp(pixel_coords[..., d], 0, self.shape[d] - 1)

        for b in range(batch_size):
            p_id = patient_idx[b].item()
            
            # Basis-Daten holen
            base_mask = self.base_masks[p_id].to(device)
            t_start = self.base_times[p_id]
            
            # Aktueller Ziel-Zeitpunkt (steht im letzten Kanal von x)
            t_current = x[b, 0, 3].item() 
            dt = max(0.0, t_current - t_start)
            
            # Berechne die Anzahl der Dilations-Schritte basierend auf der Zeit
            dilation_iterations = int(np.round(dt * self.pixels_per_day))
            
            if dilation_iterations > 0:
                # Morphologische Dilation auf der 3D-Maske durchführen
                # Da scipy nicht auf der GPU läuft, machen wir es kurz in CPU numpy
                mask_np = base_mask.cpu().numpy() > 0.5
                dilated_np = binary_dilation(mask_np, iterations=dilation_iterations)
                current_mask = torch.tensor(dilated_np, dtype=torch.float32, device=device)
            else:
                current_mask = base_mask
            
            # Mappe die kontinuierlichen Abfrage-Samples auf die dilatierte 3D-Maske
            idx_x = pixel_coords[b, :, 0]
            idx_y = pixel_coords[b, :, 1]
            idx_z = pixel_coords[b, :, 2]
            
            sampled_values = current_mask[idx_x, idx_y, idx_z]
            
            # Da dein INR Logits ausgibt (wegen BCEWithLogitsLoss), wandeln wir 
            # die binäre Maske (0 oder 1) in extreme Logits um (+-15 reicht für Sigmoid)
            logits = torch.where(sampled_values > 0.5, 15.0, -15.0)
            out[b, :, 0] = logits
            
        return out

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
            dates_list = trj.dates
            if idx in self.val_end_date_idx:
                dates_list = trj.dates[:-1]
            if hasattr(trj, 'allowed_dates'):
                dates_list = trj.allowed_dates
            random_time_point = str(np.random.choice(dates_list))
        else:
            if self.mode == 'valid_extrapolation':
                random_time_point = str(trj.dates[-1])
            if self.mode == 'valid_interpolation':
                random_time_point = str(self.val_date_idx[idx])

        labels, time_point = trj.load_labels_for_inr(selected_date=random_time_point, absolute_day_number=True)
        
        lesion_mask = labels > 0.5        
        pos_coords = np.argwhere(lesion_mask)

        border_samples = binary_dilation(labels) - labels
        border_samples_coords = np.argwhere(border_samples)

        MAX_SAMPLES = 700
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

def validate_polation(data_loader, text, global_step, criterion):
    loss = 0
    for v_coords, v_labels, v_p_idx in tqdm(data_loader, desc=text, total=len(data_loader)):
        v_coords, v_labels, v_p_idx = v_coords.to(device), v_labels.unsqueeze(-1).to(device), v_p_idx.to(device)
        v_preds = model(v_coords, v_p_idx)
        loss += criterion.dice_loss(v_preds, v_labels).item()
    mlflow.log_metric(text.lower().replace(" ", "_"), loss / len(data_loader), step=global_step)

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

    plt.rcParams['animation.convert_path'] = 'magick'
    writer = ImageMagickWriter(fps=10, extra_args=['-layers', 'Optimize'])
    file_name = f"/tmp/{epoch:04d}_epoch_lesion_heatmap_patient_{patient_idx.item()}_sample_{sample_idx}.gif"
    ani.save(file_name, writer=writer, dpi=50)

    mlflow.log_artifact(file_name)
    
    plt.close(fig)



def train_dilation(model, valid_interpolation_loader, valid_extrapolation_loader, device='cuda'):
    criterion = Loss_BCE_Dice()

    model.to(device)
    losses = []
        
    global_step = 0

    model.eval()

    with torch.no_grad():
        total_loss = 0
        for i, (coords, labels, patient_idx) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):

            coords = coords.to(device)
            labels = labels.to(device).unsqueeze(-1)
            patient_idx = patient_idx.to(device)

            if coords.shape[1] == 0:
                continue

            predictions = model(coords, patient_idx)

            loss = criterion(predictions, labels)
            
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

        torch.cuda.empty_cache()

        if epoch % 20 == 0 and epoch != 0:
            with torch.no_grad():
                visualize_samples(model, epoch, monitoring_samples, train_loader, "train")
                visualize_samples(model, epoch, monitoring_samples, valid_interpolation_loader, "valid")
                validate_polation(valid_extrapolation_loader, "Valid Extrapolation", global_step, criterion)
                validate_polation(valid_interpolation_loader, "Valid Interpolation", global_step, criterion)
                for m in monitoring_samples:
                    plot_lesion_time_evolution(model, epoch, m, train_loader)

        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    
    for m in monitoring_samples:
        plot_lesion_time_evolution(model, epoch, m, train_loader, side_length=500)

    return losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with mlflow.start_run(run_name="NaiveDilationBaseline"):
    mri_dataloader = MRI_Dataloader()
    mri_dataloader.cache_lesion_trajectories_from_n_scans(n_scans=3, only_growing=True)
    trajectories = mri_dataloader.cache_lesion_trajectories * 10

    batchsize = 1
        
    valid_interpolation_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='valid_interpolation')
    valid_interpolation_loader = DataLoader(valid_interpolation_dataset, batch_size=batchsize, shuffle=False)

    valid_extrapolation_dataset = LesionDataset(trajectories, context_radius=5, background_samples_proportion=1, device=device, mode='valid_extrapolation', val_date_idx=train_dataset.val_date_idx, val_end_date_idx=train_dataset.val_end_date_idx)
    valid_extrapolation_loader = DataLoader(valid_extrapolation_dataset, batch_size=batchsize, shuffle=False)

    model = NaiveDilationBaseline(trajectories, shape=(500, 500, 50), pixels_per_day=0.08)

    losses = train_dilation(model, valid_interpolation_loader, valid_extrapolation_loader, device=device)
    
    final_loss = losses[-1]
    mlflow.log_metric("final_train_loss", final_loss)