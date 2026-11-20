"""Patient-wise ODE training with dense, target-lesion crop validation."""

import argparse
import os

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_ode as lesion_training
from utils.dataloader_patient import (
    PatientLesionCropExtrapolationDataset,
    PatientLesionCropInterpolationDataset,
)
from utils.mri_dataloader import MRI_Dataloader


def validate_patient_polation(data_loader, text, plot, global_step, criterion):
    del criterion
    global config, model

    dice_scores = []
    volume_errors_cm3 = []
    loss_t_pairs = []
    threshold = getattr(config, "validation_prediction_threshold", 0.5)
    chunk_size = getattr(config, "validation_prediction_chunk_size", 65536)

    if chunk_size <= 0:
        raise ValueError("validation_prediction_chunk_size must be positive.")

    model.eval()
    with torch.no_grad():
        for coords, labels, patient_idx, days_from_extrapolation, voxel_volume_cm3 in tqdm(
            data_loader, desc=text, total=len(data_loader)
        ):
            coords = coords.squeeze(0)
            labels = labels.reshape(-1) > 0.5
            patient_idx = patient_idx.to(config.device)
            predicted_count = 0
            intersection_count = 0
            target_count = int(labels.sum().item())

            for start in range(0, coords.shape[0], chunk_size):
                end = min(start + chunk_size, coords.shape[0])
                chunk_coords = coords[start:end].unsqueeze(0).to(config.device)
                chunk_predictions = (
                    torch.sigmoid(model(chunk_coords, patient_idx)).reshape(-1)
                    > threshold
                ).cpu()
                chunk_labels = labels[start:end]
                predicted_count += int(chunk_predictions.sum().item())
                intersection_count += int(
                    torch.logical_and(chunk_predictions, chunk_labels).sum().item()
                )

            denominator = predicted_count + target_count
            dice_score = (
                1.0
                if denominator == 0
                else (2.0 * intersection_count) / denominator
            )
            dice_scores.append(dice_score)

            voxel_volume = float(voxel_volume_cm3.reshape(-1)[0].item())
            volume_error_cm3 = abs(predicted_count - target_count) * voxel_volume
            volume_errors_cm3.append(volume_error_cm3)
            extrapolation_days = float(days_from_extrapolation.reshape(-1)[0].item())
            loss_t_pairs.append(
                (1.0 - dice_score, torch.tensor(extrapolation_days))
            )

    if not dice_scores:
        raise ValueError(f"{text} dataset contains no validation samples.")

    metric_prefix = text.lower().replace(" ", "_")
    mean_dice = float(np.mean(dice_scores))
    volume_errors = np.asarray(volume_errors_cm3)
    mlflow.log_metrics(
        {
            f"{metric_prefix}_dice_loss": 1.0 - mean_dice,
            f"{metric_prefix}_dice_score": mean_dice,
            f"{metric_prefix}_median_wrongly_predicted_volume_cm3": float(
                np.median(volume_errors)
            ),
            f"{metric_prefix}_average_wrongly_predicted_volume_cm3": float(
                np.mean(volume_errors)
            ),
            f"{metric_prefix}_wrongly_predicted_volume_cm3": float(
                np.sum(volume_errors)
            ),
        },
        step=global_step,
    )

    if text == "Valid Extrapolation" and plot:
        lesion_training.plot_extrapolation_time(loss_t_pairs, global_step)


def train_patientwise(
    model,
    train_loader,
    valid_interpolation_loader,
    valid_extrapolation_loader,
    plotting_dataset,
    config,
):
    """Reuse the existing training loop with the patient-wise dense evaluator."""
    original_validator = lesion_training.validate_polation
    lesion_training.config = config
    lesion_training.model = model
    lesion_training.validate_polation = validate_patient_polation
    try:
        return lesion_training.train_inr(
            model,
            train_loader,
            valid_interpolation_loader,
            valid_extrapolation_loader,
            plotting_dataset,
            config,
        )
    finally:
        lesion_training.validate_polation = original_validator


def main():
    global config, model

    parser = argparse.ArgumentParser(
        description="Train and evaluate a patient-wise lesion ODE model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/06_neural_ode/config_patient_wise.py",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = lesion_training.add_base_configurations(
        lesion_training.load_config(args.config)
    )
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run(run_name=config.mlflow_run_name):
        mri_dataloader = MRI_Dataloader()
        mri_dataloader.cache_lesion_trajectories_from_n_scans(
            n_scans=config.lesion_trajectories_with_more_than_n_scans,
            only_growing=config.use_only_growing_lesions,
        )
        trajectories = mri_dataloader.cache_lesion_trajectories
        if trajectories is None:
            raise RuntimeError("No lesion trajectories were loaded.")
        training_trajectories = (
            trajectories * config.training_dataset_samples_duplication_factor
        )

        train_dataset = config.train_dataset_class(
            training_trajectories,
            dialation_iterations=config.dialation_iterations,
            device=config.device,
            background_samples=config.background_samples,
            oversample_positive_samples=config.oversample_positive_samples,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batchsize,
            shuffle=True,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
        )

        validation_kwargs = {
            "device": config.device,
            "crop_margin_mm": getattr(config, "validation_crop_margin_mm", 10.0),
        }
        valid_interpolation_dataset = PatientLesionCropInterpolationDataset(
            trajectories, **validation_kwargs
        )
        valid_extrapolation_dataset = PatientLesionCropExtrapolationDataset(
            trajectories, **validation_kwargs
        )
        valid_interpolation_loader = DataLoader(
            valid_interpolation_dataset, batch_size=1, shuffle=False
        )
        valid_extrapolation_loader = DataLoader(
            valid_extrapolation_dataset, batch_size=1, shuffle=False
        )

        plotting_dataset_class = getattr(
            config, "plotting_dataset_class", None
        )
        plotting_dataset = (
            plotting_dataset_class(trajectories.copy())
            if plotting_dataset_class is not None
            else []
        )

        max_embedding_id = max(
            train_dataset.get_embedding_id(trj) for trj in trajectories
        )
        model_params = dict(config.model_params)
        model_params.setdefault("num_patients", int(max_embedding_id) + 1)
        model = config.model(trajectories=training_trajectories, **model_params)

        mlflow.log_params(
            {
                "dataset_size": len(train_dataset),
                "loss_function": type(config.loss_fn).__name__,
                "validation_crop_margin_mm": validation_kwargs["crop_margin_mm"],
                **{
                    key: getattr(config, key)
                    for key in dir(config)
                    if not key.startswith("_")
                },
            }
        )
        mlflow.log_artifact(args.config)
        mlflow.log_artifact(os.path.basename(__file__))

        losses = train_patientwise(
            model,
            train_loader,
            valid_interpolation_loader,
            valid_extrapolation_loader,
            plotting_dataset,
            config,
        )
        if losses:
            mlflow.log_metric("final_train_loss", losses[-1])


if __name__ == "__main__":
    main()
