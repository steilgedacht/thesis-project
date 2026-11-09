import mlflow
import torch
import json
import argparse
import matplotlib
import importlib.util
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mlflow.pytorch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.mri_dataloader import MRI_Dataloader
from utils.dataloader import LesionDataset, Plotting_LesionDataset
from utils.train_plotting import plot_lesion_time_evolution, plot_contanct_sheet
from utils.loss_bce_dice import Loss_BCE_Dice

mlflow.set_tracking_uri("http://127.0.0.1:5124")
mlflow.set_experiment("Lesion_Embedding")

def load_config(config_path: str):
    config_file = Path(config_path)
    spec = importlib.util.spec_from_file_location(config_file.stem, config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()


def add_base_configurations(config):
    with open("configs/base_config.json", "r") as f:
        base_config = json.load(f)
    for key, value in base_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    return config


def _dataset_item_to_trajectory_and_date(item):
    """Return the trajectory object and the date used for that dataset item."""
    if isinstance(item, tuple) and len(item) == 3:
        _, date, trj = item
        return trj, str(date)
    return item, None


def _voxel_volume_cm3_from_trajectory(trj, selected_date=None):
    """Compute the physical voxel volume for the given trajectory at one date."""
    if selected_date is None:
        selected_date = str(trj.allowed_dates[0] if hasattr(trj, 'allowed_dates') else trj.dates[0])
    _, affine = trj.load_labels_for_inr(selected_date=selected_date, absolute_day_number=True, affine=True)
    voxel_volume_mm3 = float(np.prod(np.abs(np.diag(affine)[:3])))
    return voxel_volume_mm3 / 1000.0


def _project_onto_simplex(values):
    """Project values onto the non-negative vectors whose entries sum to one."""
    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(1, values.numel() + 1, device=values.device, dtype=values.dtype)
    active = sorted_values - (cumulative - 1) / indices > 0
    last_active = torch.nonzero(active, as_tuple=False)[-1, 0]
    threshold = (cumulative[last_active] - 1) / (last_active + 1)
    return torch.clamp(values - threshold, min=0)


def validate_polation(data_loader, text, global_step, criterion):
    loss = 0.0
    volume_error_cm3 = 0.0

    batch_size = getattr(data_loader, 'batch_size', 1) or 1

    for batch_idx, (v_coords, v_labels, v_p_idx) in enumerate(tqdm(data_loader, desc=text, total=len(data_loader))):
        v_coords, v_labels, v_p_idx = v_coords.to(config.device), v_labels.unsqueeze(-1).to(config.device), v_p_idx.to(config.device)
        v_p_idx = torch.zeros_like(v_p_idx)
        v_preds = model(v_coords, v_p_idx)
        l = criterion.dice_loss(v_preds, v_labels).item()
        loss += l
        dataset_start = batch_idx * batch_size
        dataset_end = dataset_start + v_coords.shape[0]
        for local_idx, sample_p_idx in enumerate(v_p_idx.cpu().tolist()):
            data_idx = dataset_start + local_idx
            if data_idx >= len(data_loader.dataset.trajectories):
                break
            dataset_item = data_loader.dataset.trajectories[data_idx]
            trj, date = _dataset_item_to_trajectory_and_date(dataset_item)
            voxel_volume_cm3 = _voxel_volume_cm3_from_trajectory(trj, date)

            pred_count = (torch.sigmoid(v_preds[local_idx]).squeeze(-1) > 0.5).sum().item()
            true_count = (v_labels[local_idx].squeeze(-1) > 0.5).sum().item()
            volume_error_cm3 += abs(pred_count - true_count) * voxel_volume_cm3

    mlflow.log_metrics({
        text.lower().replace(" ", "_") + "_dice_loss"  : loss / len(data_loader),
        text.lower().replace(" ", "_") + "_dice_score" : 1 - loss / len(data_loader),
        text.lower().replace(" ", "_") + "_wrongly_predicted_volume_cm3" : volume_error_cm3,
    }, step=global_step)



def fit_trajectory_to_model(
    model,
    train_loader,
    plotting_LesionDataset,
    config,
    epochs=100,
    lr=1e-3,
    device='cuda',
    combination_epochs=30,
):
    model.to(device)
    losses = []

    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    if combination_epochs < 0:
        raise ValueError("combination_epochs must be non-negative")

    combination_phase_epochs = min(combination_epochs, epochs)
    direct_phase_epochs = epochs - combination_phase_epochs
    original_requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    for param in model.parameters():
        param.requires_grad_(False)

    basis_embeddings = model.patient_embedding.weight.detach().clone()
    coefficients = torch.nn.Parameter(torch.zeros(basis_embeddings.shape[0], device=device))
    with torch.no_grad():
        coefficients[0] = 1.0
    criterion = Loss_BCE_Dice()

    global_step = 0

    def train_embedding_phase(phase_name, phase_epochs, optimizer, embedding_for_batch, project_coefficients=False):
        nonlocal global_step
        phase_losses = []
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25)

        def use_fitted_embedding(_embedding, _inputs, output):
            return embedding_for_batch(output)

        embedding_hook = model.patient_embedding.register_forward_hook(use_fitted_embedding)
        try:
            for phase_epoch in range(phase_epochs):
                total_loss = 0.0
                model.train()
                for i, (coords, labels, patient_idx) in tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"{phase_name} epoch {phase_epoch + 1}/{phase_epochs}",
                ):
                    coords = coords.to(device)
                    labels = labels.to(device).unsqueeze(-1)
                    patient_idx = torch.zeros_like(patient_idx, device=device)

                    if coords.shape[1] == 0:
                        continue

                    optimizer.zero_grad()
                    predictions = model(coords, patient_idx)
                    loss = criterion(predictions, labels)
                    loss.backward()
                    optimizer.step()
                    if project_coefficients:
                        with torch.no_grad():
                            coefficients.copy_(_project_onto_simplex(coefficients))

                    total_loss += loss.item()
                    scheduler.step(metrics=loss.item())

                    step_metrics = {
                        "training_loss": loss.item(),
                        "training_loss_bce": criterion.loss_bce.item(),
                        "training_loss_dice": criterion.loss_dice.item(),
                        "number_of_correctly_predicted_1_labels": (
                            (predictions > 0.5).sum().item()
                            / (labels > 0.5).sum().item()
                        ),
                        f"{phase_name}_training_loss": loss.item(),
                    }
                    mlflow.log_metrics(step_metrics, step=global_step)
                    global_step += 1
                    mlflow.log_metric(
                        "learning_rate",
                        optimizer.param_groups[0]['lr'],
                        step=global_step,
                    )

                    if i % 10 == 0 and torch.device(device).type == "cuda":
                        torch.cuda.empty_cache()

                model.eval()
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()

                avg_loss = total_loss / max(len(train_loader), 1)
                phase_losses.append(avg_loss)
                if (phase_epoch + 1) % 10 == 0:
                    print(
                        f"{phase_name} epoch {phase_epoch + 1}/{phase_epochs}, "
                        f"Loss: {avg_loss:.6f}"
                    )
        finally:
            embedding_hook.remove()
        return phase_losses

    try:
        combination_optimizer = optim.AdamW(
            [coefficients],
            lr=lr,
            weight_decay=1e-6,
        )
        losses.extend(
            train_embedding_phase(
                "linear_combination",
                combination_phase_epochs,
                combination_optimizer,
                lambda output: (coefficients @ basis_embeddings)
                .unsqueeze(0)
                .expand_as(output),
                project_coefficients=True,
            )
        )

        with torch.no_grad():
            fitted_embedding = coefficients @ basis_embeddings
            model.patient_embedding.weight[0].copy_(fitted_embedding)
            mlflow.log_metric("coefficient_sum", coefficients.sum().item())
            mlflow.log_metric("maximum_coefficient", coefficients.max().item())
            mlflow.log_dict(
                {
                    "basis_indices": list(range(coefficients.numel())),
                    "coefficients": coefficients.detach().cpu().tolist(),
                },
                "new_patient_embedding_coefficients.json",
            )

        if combination_phase_epochs > 0:
            plotting_LesionDataset.trajectories = [
                mri_dataloader.find_lesion_trajectory('YG_WCONBI42RLEZ', 1)
            ]
            trj = next(iter(plotting_LesionDataset))
            trj.embedding_id = np.int64(0)
            plot_lesion_time_evolution(
                model,
                combination_phase_epochs - 1,
                trj,
                config,
                final_side_length=True,
            )

        direct_embedding = torch.nn.Parameter(fitted_embedding.detach().clone())
        direct_optimizer = optim.AdamW(
            [direct_embedding],
            lr=lr*0.01,
            weight_decay=1e-6,
        )
        losses.extend(
            train_embedding_phase(
                "direct_embedding",
                direct_phase_epochs,
                direct_optimizer,
                lambda output: direct_embedding.unsqueeze(0).expand_as(output),
            )
        )

        with torch.no_grad():
            model.patient_embedding.weight[0].copy_(direct_embedding)
            plotting_LesionDataset.trajectories = [mri_dataloader.find_lesion_trajectory('YG_WCONBI42RLEZ', 1)]
            trj = next(iter(plotting_LesionDataset))
            trj.embedding_id = np.int64(0)
            validate_polation(train_loader, "Valid Interpolation", global_step, criterion)
            final_epoch = max(epochs - 1, 0)
            plot_contanct_sheet(model, final_epoch, trj, config, final_side_length=True)
            plot_lesion_time_evolution(model, final_epoch, trj, config, final_side_length=True)

        return losses
    finally:
        for parameter, requires_grad in original_requires_grad.items():
            parameter.requires_grad_(requires_grad)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with mlflow.start_run(run_name="embedding learning", tags={"type" : "new_patient_fit"}):
    argument_parser = argparse.ArgumentParser(description="Train a Lesion Trajectory model.")
    argument_parser.add_argument("--config", type=str, default="configs/00_default/config.py", help="Path to the configuration file.")
    args = argument_parser.parse_args()
    config = load_config(args.config)
    config = add_base_configurations(config)

    mri_dataloader = MRI_Dataloader()
    trj = [mri_dataloader.find_lesion_trajectory('YG_WCONBI42RLEZ', 1)] * 20
    
    train_dataset = LesionDataset(trj, device=device)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=7, prefetch_factor=2, pin_memory=True, persistent_workers=True)
    plotting_LesionDataset = Plotting_LesionDataset(trj)

    run_id = "f82c0dc523134b0699c17c7b16e7a3d7"
    model_uri = f"runs:/{run_id}/lesion_ode_model"
    model = mlflow.pytorch.load_model(model_uri)

    losses = fit_trajectory_to_model(model, train_loader, plotting_LesionDataset, config=config, epochs=100, lr=1., device=device)
