"""Compare saved MLflow INR, Neural ODE, and LSTM models on one uniform grid.

The evaluator uses 500 points per axis by default (125 million points per
held-out scan). Run it from the repository root, for example:

    python validation_arena.py \
        --inr-run-id <RUN_ID> \
        --node-run-id <RUN_ID> \
        --lstm-run-id <RUN_ID>

The uniform query cube is [-1, 1]^3 in the INR/Neural-ODE coordinate system.
Ground-truth masks are sampled at the corresponding physical locations, and
LSTM probabilities are interpolated from the model's native grid to those
same locations. Results are written per held-out scan to CSV.

Unless --skip-embedding-test is set, each model also fits one embedding on
YG_WCONBI42RLEZ lesion 1 and reports fitted-embedding Dice for every timepoint.
"""

import argparse
import ast
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from mlflow.tracking import MlflowClient
from scipy.ndimage import map_coordinates

from utils.dataloader import LesionDataset
from utils.dataloader_lstm import LesionSequenceDataset
from utils.loss_bce_dice import Loss_BCE_Dice
from utils.model_inr import LesionINR
from utils.model_lstm import LesionLSTM, LesionLatentLSTM
from utils import model_neural_ode_validation
from utils import model_neural_ode_translation_validation
from utils.mri_dataloader import MRI_Dataloader
from utils.paths import DatasetPaths


MODEL_CLASSES = {
    "inr": (LesionINR,),
    "node": (
        model_neural_ode_validation.NeuralODE_INR,
        model_neural_ode_translation_validation.NeuralODETranslationINR,
    ),
    "lstm": (LesionLSTM, LesionLatentLSTM),
}
EMBEDDING_TRAJECTORY = ("YG_WCONBI42RLEZ", 1)


def _print_progress(iterable, description, unit, interval_seconds=30.0):
    try:
        total = len(iterable)
    except TypeError:
        total = None

    start_time = time.monotonic()
    next_report = start_time + interval_seconds
    completed = 0
    print(f"{description}: started", file=sys.stderr, flush=True)
    for item in iterable:
        yield item
        completed += 1
        now = time.monotonic()
        if now >= next_report:
            status = f"{completed}/{total}" if total is not None else str(completed)
            if total:
                status += f" ({completed / total:.0%})"
            elapsed_minutes = (now - start_time) / 60.0
            print(
                f"{description}: {status} {unit}, "
                f"{elapsed_minutes:.1f} min elapsed",
                file=sys.stderr,
                flush=True,
            )
            next_report = now + interval_seconds

    elapsed_minutes = (time.monotonic() - start_time) / 60.0
    status = f"{completed}/{total}" if total is not None else str(completed)
    print(
        f"{description}: complete ({status} {unit}, {elapsed_minutes:.1f} min)",
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True)
class EvaluationCase:
    lesion_id: str
    trajectory: Any
    split: str
    target_date: str
    days_from_endpoint: int | float | None
    history_dates: tuple[str, ...]


@dataclass(frozen=True)
class TargetVolume:
    labels: np.ndarray
    time_point: float
    center_phys: np.ndarray
    inverse_affine: np.ndarray


@dataclass
class LoadedModel:
    family: str
    run_id: str
    model: torch.nn.Module
    params: dict[str, str]


def _literal_param(params: dict[str, str], key: str, default: Any) -> Any:
    value = params.get(key)
    if value is None:
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _find_model_artifacts(client: MlflowClient, run_id: str) -> list[str]:
    found = []
    pending = [("", 0)]
    while pending:
        path, depth = pending.pop()
        if depth > 5:
            continue
        for artifact in client.list_artifacts(run_id, path):
            if artifact.is_dir:
                pending.append((artifact.path, depth + 1))
            elif Path(artifact.path).name == "MLmodel":
                found.append(str(Path(artifact.path).parent))
    return sorted(set(found))


def _load_validation_model(model_uri: str, device: torch.device) -> torch.nn.Module:
    validation_modules = {
        "utils.model_neural_ode": model_neural_ode_validation,
        "utils.model_neural_ode_translation": model_neural_ode_translation_validation,
    }
    original_modules = {
        name: sys.modules.get(name) for name in validation_modules
    }
    sys.modules.update(validation_modules)
    try:
        return mlflow.pytorch.load_model(model_uri, map_location=device)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_model(
    client: MlflowClient,
    family: str,
    run_id: str,
    artifact: str | None,
    device: torch.device,
) -> LoadedModel:
    run = client.get_run(run_id)
    params = dict(run.data.params)
    model_path = artifact or params.get("model_save_name")
    if model_path is None:
        candidates = _find_model_artifacts(client, run_id)
        if len(candidates) != 1:
            raise ValueError(
                f"Run {run_id} has {len(candidates)} MLflow model artifacts. "
                f"Specify --{family}-artifact explicitly. Found: {candidates}"
            )
        model_path = candidates[0]

    model_uri = f"runs:/{run_id}/{model_path}" if model_path else f"runs:/{run_id}"
    model = _load_validation_model(model_uri, device)
    if not isinstance(model, MODEL_CLASSES[family]):
        expected = ", ".join(cls.__name__ for cls in MODEL_CLASSES[family])
        raise TypeError(
            f"Run {run_id} artifact {model_path!r} loaded as "
            f"{type(model).__name__}; expected {family} model ({expected})."
        )
    model.to(device).eval()
    return LoadedModel(family, run_id, model, params)


def _selected_trajectories(
    dataloader: MRI_Dataloader, params: dict[str, str]
) -> dict[str, Any]:
    min_scans = int(
        _literal_param(params, "lesion_trajectories_with_more_than_n_scans", 6)
    )
    only_growing = bool(_literal_param(params, "use_only_growing_lesions", True))
    dataloader.cache_lesion_trajectories_from_n_scans(
        n_scans=min_scans, only_growing=only_growing
    )
    assert dataloader.cache_lesion_trajectories is not None
    return {
        f"{trajectory.patient_id}_{trajectory.label_id}": trajectory
        for trajectory in dataloader.cache_lesion_trajectories
    }


def _validation_samples() -> dict[str, dict[str, Any]]:
    with open(DatasetPaths().validation_samples(), encoding="utf-8") as file:
        samples = json.load(file)
    return {
        f"{sample['Patient']}_{sample['Lesion']}": sample for sample in samples
    }


def _load_target_volume(case: EvaluationCase) -> TargetVolume:
    (labels, time_point), affine = case.trajectory.load_labels_for_inr(
        selected_date=case.target_date,
        absolute_day_number=True,
        affine=True,
    )
    labels = np.asarray(labels, dtype=np.float32)
    voxel_center = (np.asarray(labels.shape, dtype=np.float64) - 1.0) / 2.0
    center_phys = affine[:3, :3] @ voxel_center + affine[:3, 3]
    return TargetVolume(
        labels=labels,
        time_point=float(time_point),
        center_phys=center_phys,
        inverse_affine=np.linalg.inv(affine),
    )


def _make_cases(
    shared_trajectories: dict[str, Any],
    validation_samples: dict[str, dict[str, Any]],
    split: str,
) -> list[EvaluationCase]:
    cases = []
    for lesion_id, trajectory in sorted(shared_trajectories.items()):
        if lesion_id not in validation_samples:
            continue
        dates = list(
            trajectory.allowed_dates
            if hasattr(trajectory, "allowed_dates")
            else trajectory.dates
        )
        sample = validation_samples[lesion_id]
        if split in ("both", "interpolation"):
            target_index = int(sample["interpolation"])
            target_date = str(dates[target_index])
            history = tuple(str(date) for date in dates[:target_index])
            if not history:
                history = (target_date,)
            cases.append(
                EvaluationCase(
                    lesion_id, trajectory, "interpolation", target_date, None, history
                )
            )
        if split in ("both", "extrapolation"):
            held_out_dates = sample["extrapolation_dates"]
            held_out_steps = sample["extrapolation_T_minus"]
            history = tuple(
                str(date) for date in dates if date not in held_out_dates
            )
            for step, date in zip(held_out_steps, held_out_dates):
                cases.append(
                    EvaluationCase(
                        lesion_id,
                        trajectory,
                        "extrapolation",
                        str(date),
                        float(step),
                        history,
                    )
                )
    return cases


def _coordinates_and_labels(
    target_volume: TargetVolume,
    start: int,
    end: int,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one uniform-cube coordinate chunk and sample its binary target."""
    flat_indices = np.arange(start, end, dtype=np.int64)
    plane_size = resolution * resolution
    index_0 = flat_indices // plane_size
    index_1 = (flat_indices // resolution) % resolution
    index_2 = flat_indices % resolution
    scale = 2.0 / (resolution - 1)
    xyz = np.stack(
        (
            -1.0 + index_0 * scale,
            -1.0 + index_1 * scale,
            -1.0 + index_2 * scale,
        ),
        axis=1,
    ).astype(np.float32)

    physical_coords = target_volume.center_phys + xyz.astype(np.float64) * 100.0
    voxel_coords = (
        target_volume.inverse_affine[:3, :3] @ physical_coords.T
        + target_volume.inverse_affine[:3, 3:4]
    )
    target = map_coordinates(
        target_volume.labels,
        voxel_coords,
        order=0,
        mode="constant",
        cval=0.0,
    ) > 0.5

    model_coords = np.concatenate(
        (
            xyz,
            np.full(
                (len(xyz), 1), target_volume.time_point, dtype=np.float32
            ),
        ),
        axis=1,
    )
    return model_coords, target


def _dice_sums(
    sums: dict[str, float],
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> None:
    probabilities = np.clip(probabilities.astype(np.float64, copy=False), 0.0, 1.0)
    targets = targets.astype(bool, copy=False)
    sums["probability"] += float(probabilities.sum())
    sums["target"] += float(targets.sum())
    sums["soft_intersection"] += float(probabilities[targets].sum())
    predicted = probabilities >= 0.5
    sums["hard_prediction"] += int(predicted.sum())
    sums["hard_intersection"] += int(np.logical_and(predicted, targets).sum())


def _dice_results(sums: dict[str, float]) -> tuple[float, float]:
    soft = (
        2.0 * sums["soft_intersection"] + 1e-6
    ) / (sums["probability"] + sums["target"] + 1e-6)
    hard = (
        2.0 * sums["hard_intersection"] + 1e-6
    ) / (sums["hard_prediction"] + sums["target"] + 1e-6)
    return soft, hard


def _summary_metrics(rows: list[dict[str, Any]], model_name: str) -> dict[str, float]:
    metrics = {}
    for split in ("interpolation", "extrapolation", "trajectory"):
        split_rows = [
            row
            for row in rows
            if row["model"] == model_name and row["split"] == split
        ]
        if not split_rows:
            continue
        prefix = f"{model_name}_{split}"
        dice_loss = float(np.mean([1.0 - row["soft_dice"] for row in split_rows]))
        volume_errors = [
            row["wrongly_predicted_volume_cm3"] for row in split_rows
        ]
        metrics.update(
            {
                f"{prefix}_dice_loss": dice_loss,
                f"{prefix}_dice_score": 1.0 - dice_loss,
                f"{prefix}_median_wrongly_predicted_volume_cm3": float(
                    np.median(volume_errors)
                ),
                f"{prefix}_average_wrongly_predicted_volume_cm3": float(
                    np.mean(volume_errors)
                ),
                f"{prefix}_wrongly_predicted_volume_cm3": float(
                    np.sum(volume_errors)
                ),
                f"{prefix}_hard_dice_score": float(
                    np.mean([row["hard_dice_at_0.5"] for row in split_rows])
                ),
            }
        )
    return metrics


def _project_onto_simplex(values: torch.Tensor) -> torch.Tensor:
    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(
        1, values.numel() + 1, device=values.device, dtype=values.dtype
    )
    active = sorted_values - (cumulative - 1) / indices > 0
    last_active = torch.nonzero(active, as_tuple=False)[-1, 0]
    threshold = (cumulative[last_active] - 1) / (last_active + 1)
    return torch.clamp(values - threshold, min=0)


def _infer_continuous_model(
    loaded: LoadedModel,
    target_volume: TargetVolume,
    patient_idx: int,
    resolution: int,
    chunk_size: int,
    device: torch.device,
    embedding_override: torch.Tensor | None = None,
) -> dict[str, float]:
    count = resolution**3
    sums = {
        "probability": 0.0,
        "target": 0.0,
        "soft_intersection": 0.0,
        "hard_prediction": 0.0,
        "hard_intersection": 0.0,
    }
    patient_tensor = torch.tensor([patient_idx], dtype=torch.long, device=device)

    hook = (
        _embedding_override(loaded.model, embedding_override)
        if embedding_override is not None
        else None
    )
    try:
        with torch.inference_mode():
            for start in _print_progress(
                range(0, count, chunk_size),
                f"{loaded.family} {target_volume.time_point:g}d grid",
                "chunks",
            ):
                end = min(start + chunk_size, count)
                coords, targets = _coordinates_and_labels(
                    target_volume, start, end, resolution
                )
                coords_tensor = torch.from_numpy(coords).unsqueeze(0).to(device)
                logits = loaded.model(coords_tensor, patient_tensor).reshape(-1)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                _dice_sums(sums, probabilities, targets)
    finally:
        if hook is not None:
            hook.remove()
    return sums


def _lstm_dataset(
    trajectories: list[Any],
    loaded: LoadedModel,
) -> LesionSequenceDataset:
    model = loaded.model
    grid_size = getattr(model, "grid_size", None)
    if grid_size is None:
        grid_size = model.decoder.out_size
    dataset_params = _literal_param(loaded.params, "dataset_params", {})
    if not isinstance(dataset_params, dict):
        dataset_params = {}
    return LesionSequenceDataset(
        trajectories,
        grid_size=tuple(grid_size),
        margin_mm=float(dataset_params.get("margin_mm", 20.0)),
        min_extent_mm=float(dataset_params.get("min_extent_mm", 80.0)),
        max_sequence_len=int(dataset_params.get("max_sequence_len", 6)),
    )


def _infer_lstm(
    loaded: LoadedModel,
    case: EvaluationCase,
    target_volume: TargetVolume,
    trajectory_index: int,
    sequence_dataset: LesionSequenceDataset,
    resolution: int,
    chunk_size: int,
    device: torch.device,
    embedding_override: torch.Tensor | None = None,
) -> dict[str, float]:
    if not case.history_dates:
        raise ValueError(
            f"No observed history for {case.lesion_id} at {case.target_date}."
        )
    dates_needed = list(dict.fromkeys((*case.history_dates, case.target_date)))
    resampled = sequence_dataset._compute_resampled_trajectory(
        trajectory_index, dates_needed
    )
    history_grids = np.stack(
        [resampled["grids"][date] for date in case.history_dates], axis=0
    )[:, None]
    history_times = np.asarray(
        [resampled["times"][date] for date in case.history_dates],
        dtype=np.float32,
    )
    target_time = float(resampled["times"][case.target_date])
    model = loaded.model
    patient_idx = sequence_dataset.get_embedding_id(case.trajectory)
    grids_tensor = torch.from_numpy(history_grids).unsqueeze(0).to(device)
    times_tensor = torch.from_numpy(
        np.concatenate((history_times, np.asarray([target_time], dtype=np.float32)))
    ).unsqueeze(0).to(device)
    patient_tensor = torch.tensor([patient_idx], dtype=torch.long, device=device)

    hook = (
        _embedding_override(model, embedding_override)
        if embedding_override is not None
        else None
    )
    try:
        with torch.inference_mode():
            logits = model(
                grids_tensor,
                times_tensor,
                patient_idx=patient_tensor,
                n_future=1,
            )[:, -1]
    finally:
        if hook is not None:
            hook.remove()

    extent = float(resampled["extent"])
    center = np.asarray(resampled["center"], dtype=np.float64)
    sums = {
        "probability": 0.0,
        "target": 0.0,
        "soft_intersection": 0.0,
        "hard_prediction": 0.0,
        "hard_intersection": 0.0,
    }
    count = resolution**3
    for start in _print_progress(
        range(0, count, chunk_size),
        f"LSTM {case.target_date} grid",
        "chunks",
    ):
        end = min(start + chunk_size, count)
        coords, targets = _coordinates_and_labels(
            target_volume, start, end, resolution
        )
        physical_coords = center + coords[:, :3].astype(np.float64) * 100.0
        normalized_axes = 2.0 * (physical_coords - center) / extent
        in_bounds = np.all(np.abs(normalized_axes) <= 1.0, axis=1)
        sample_grid = np.stack(
            (
                normalized_axes[:, 2],
                normalized_axes[:, 1],
                normalized_axes[:, 0],
            ),
            axis=1,
        ).astype(np.float32)
        sample_grid = torch.from_numpy(sample_grid).to(device).view(
            1, 1, 1, -1, 3
        )
        with torch.inference_mode():
            sampled_probabilities = F.grid_sample(
                torch.sigmoid(logits),
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).reshape(-1)
            probabilities = sampled_probabilities.cpu().numpy()
        probabilities[~in_bounds] = 0.0
        _dice_sums(sums, probabilities, targets)
    return sums


def _embedding_module(model: torch.nn.Module) -> torch.nn.Embedding:
    for name in ("patient_embedding", "latent_vectors", "patient_emb"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Embedding):
            return module
    raise TypeError(
        f"{type(model).__name__} has no supported patient embedding table; "
        "embedding-only adaptation cannot be run for this checkpoint."
    )


def _embedding_override(model: torch.nn.Module, embedding: torch.Tensor):
    module = _embedding_module(model)

    def replace_embedding(_module, _inputs, output):
        return embedding.unsqueeze(0).expand_as(output)

    return module.register_forward_hook(replace_embedding)


def _prepare_continuous_support(
    loaded: LoadedModel,
    case: EvaluationCase,
    samples_per_epoch: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not case.history_dates:
        raise ValueError(f"No adaptation scans available for {case.lesion_id}.")
    # Keep adaptation patient-specific: every support sample comes from this
    # case's trajectory, never from another validation patient.
    dataset = LesionDataset(
        [case.trajectory] * samples_per_epoch,
        device="cpu",
        dialation_iterations=int(
            _literal_param(loaded.params, "dialation_iterations", 1)
        ),
        background_samples=int(
            _literal_param(loaded.params, "background_samples", 3000)
        ),
        oversample_positive_samples=bool(
            _literal_param(loaded.params, "oversample_positive_samples", False)
        ),
    )
    samples = []
    for _ in range(samples_per_epoch):
        date = random.choice(case.history_dates)
        coords, labels = dataset.prepare_data(date, case.trajectory)
        samples.append(
            (
                torch.as_tensor(coords, dtype=torch.float32),
                torch.as_tensor(labels, dtype=torch.float32).unsqueeze(-1),
            )
        )
    return samples


def _prepare_lstm_support(
    loaded: LoadedModel,
    case: EvaluationCase,
    trajectory_index: int,
    sequence_dataset: LesionSequenceDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(case.history_dates) < 2:
        raise ValueError(
            f"LSTM embedding adaptation for {case.lesion_id} requires at least "
            "two observed history scans."
        )
    resampled = sequence_dataset._compute_resampled_trajectory(
        trajectory_index, list(case.history_dates)
    )
    grids = np.stack(
        [resampled["grids"][date] for date in case.history_dates], axis=0
    )[:, None]
    times = np.asarray(
        [resampled["times"][date] for date in case.history_dates],
        dtype=np.float32,
    )
    return torch.from_numpy(grids).unsqueeze(0), torch.from_numpy(times).unsqueeze(0)


def _fit_embedding(
    loaded: LoadedModel,
    case: EvaluationCase,
    trajectory_index: int,
    lstm_dataset: LesionSequenceDataset | None,
    epochs: int,
    combination_epochs: int,
    learning_rate: float,
    batch_size: int,
    samples_per_epoch: int,
    device: torch.device,
) -> torch.Tensor:
    if epochs < 1:
        raise ValueError("--embedding-epochs must be positive.")
    if combination_epochs < 0:
        raise ValueError("--embedding-combination-epochs cannot be negative.")
    if batch_size < 1 or samples_per_epoch < 1:
        raise ValueError("Embedding batch size and samples per epoch must be positive.")

    model = loaded.model
    embedding_layer = _embedding_module(model)
    original_training_mode = model.training
    model.eval()
    original_requires_grad = {
        parameter: parameter.requires_grad for parameter in model.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    recurrent_module = getattr(model, "rnn", None)
    if isinstance(recurrent_module, torch.nn.RNNBase):
        recurrent_module.train()

    basis_embeddings = embedding_layer.weight.detach().clone()
    coefficients = torch.nn.Parameter(
        torch.zeros(basis_embeddings.shape[0], device=device)
    )
    with torch.no_grad():
        coefficients[0] = 1.0
    direct_embedding = None
    criterion = Loss_BCE_Dice()
    phase_epochs = min(combination_epochs, epochs)
    support_samples = None
    support_sequence = None

    def train_phase(
        phase_name: str,
        phase_length: int,
        parameter: torch.nn.Parameter,
        optimizer: optim.Optimizer,
        project_coefficients: bool,
    ) -> None:
        if phase_length == 0:
            return

        def embedding_for_output(output):
            if project_coefficients:
                vector = coefficients @ basis_embeddings
            else:
                vector = parameter
            return vector.unsqueeze(0).expand_as(output)

        def hook(_module, _inputs, output):
            return embedding_for_output(output)

        embedding_hook = embedding_layer.register_forward_hook(hook)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=25
        )
        try:
            for _epoch in _print_progress(
                range(phase_length),
                f"{loaded.family} embedding {phase_name}",
                "epochs",
            ):
                if loaded.family == "lstm":
                    assert support_sequence is not None
                    grids, times = (
                        tensor.to(device) for tensor in support_sequence
                    )
                    patient_idx = torch.zeros(1, dtype=torch.long, device=device)
                    predictions = model(
                        grids, times, patient_idx=patient_idx, n_future=0
                    )
                    targets = grids[:, 1:]
                    if predictions.shape != targets.shape:
                        raise ValueError(
                            f"LSTM prediction shape {tuple(predictions.shape)} "
                            f"does not match adaptation targets "
                            f"{tuple(targets.shape)}."
                        )
                    optimizer.zero_grad()
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(loss.item())
                    if project_coefficients:
                        with torch.no_grad():
                            coefficients.copy_(
                                _project_onto_simplex(coefficients)
                            )
                else:
                    assert support_samples is not None
                    shuffled = random.sample(support_samples, len(support_samples))
                    for start in range(0, len(shuffled), batch_size):
                        batch = shuffled[start : start + batch_size]
                        coords = torch.stack([item[0] for item in batch]).to(device)
                        labels = torch.stack([item[1] for item in batch]).to(device)
                        patient_idx = torch.zeros(
                            len(batch), dtype=torch.long, device=device
                        )
                        optimizer.zero_grad()
                        predictions = model(coords, patient_idx)
                        loss = criterion(predictions, labels)
                        loss.backward()
                        optimizer.step()
                        scheduler.step(loss.item())
                        if project_coefficients:
                            with torch.no_grad():
                                coefficients.copy_(
                                    _project_onto_simplex(coefficients)
                                )
        finally:
            embedding_hook.remove()

    try:
        if loaded.family == "lstm":
            if lstm_dataset is None:
                raise ValueError("LSTM support dataset was not initialized.")
            support_sequence = _prepare_lstm_support(
                loaded, case, trajectory_index, lstm_dataset
            )
        else:
            support_samples = _prepare_continuous_support(
                loaded, case, samples_per_epoch
            )

        combination_optimizer = optim.AdamW(
            [coefficients], lr=learning_rate, weight_decay=1e-6
        )
        train_phase(
            "linear_combination",
            phase_epochs,
            coefficients,
            combination_optimizer,
            project_coefficients=True,
        )
        with torch.no_grad():
            fitted_embedding = coefficients @ basis_embeddings

        direct_embedding = torch.nn.Parameter(fitted_embedding.detach().clone())
        direct_optimizer = optim.AdamW(
            [direct_embedding], lr=learning_rate * 0.01, weight_decay=1e-6
        )
        train_phase(
            "direct_embedding",
            epochs - phase_epochs,
            direct_embedding,
            direct_optimizer,
            project_coefficients=False,
        )
        return direct_embedding.detach().clone()
    finally:
        for parameter, requires_grad in original_requires_grad.items():
            parameter.requires_grad_(requires_grad)
        model.train(original_training_mode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MLflow INR, Neural ODE, and/or LSTM runs on a shared "
            "uniform spatial grid."
        )
    )
    for family in MODEL_CLASSES:
        parser.add_argument(f"--{family}-run-id", help=f"MLflow {family} run ID")
        parser.add_argument(
            f"--{family}-artifact",
            help=f"Model artifact path in the {family} run (auto-detected if omitted)",
        )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5124",
        help="MLflow tracking URI (default: http://127.0.0.1:5124)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=500,
        help="Number of uniformly spaced points per cube axis (default: 500)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=65536,
        help="Query points processed per inference chunk (default: 65536)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference",
    )
    parser.add_argument(
        "--split",
        choices=("both", "interpolation", "extrapolation"),
        default="both",
        help="Which held-out validation targets to evaluate",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Optional limit on the number of validation targets, for smoke tests",
    )
    parser.add_argument(
        "--skip-embedding-test",
        action="store_true",
        help="Evaluate only the original saved embeddings, without adaptation.",
    )
    parser.add_argument(
        "--embedding-epochs",
        type=int,
        default=100,
        help="Total embedding adaptation epochs (default: 100).",
    )
    parser.add_argument(
        "--embedding-combination-epochs",
        type=int,
        default=30,
        help="Linear-combination phase epochs (default: 30).",
    )
    parser.add_argument(
        "--embedding-lr",
        type=float,
        default=1.0,
        help="First-phase learning rate; second phase uses 1%% of this (default: 1.0).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=5,
        help="Continuous-model adaptation batch size (default: 5).",
    )
    parser.add_argument(
        "--embedding-samples-per-epoch",
        type=int,
        default=20,
        help="Sampled support batches per continuous-model epoch (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_arena_results.csv"),
        help="Per-target results CSV path",
    )
    args = parser.parse_args()
    if not any(getattr(args, f"{family}_run_id") for family in MODEL_CLASSES):
        parser.error("Provide at least one of --inr-run-id, --node-run-id, or --lstm-run-id.")
    if args.resolution < 2:
        parser.error("--resolution must be at least 2.")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive.")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive.")
    if args.embedding_epochs < 1:
        parser.error("--embedding-epochs must be positive.")
    if args.embedding_combination_epochs < 0:
        parser.error("--embedding-combination-epochs cannot be negative.")
    if args.embedding_lr <= 0:
        parser.error("--embedding-lr must be positive.")
    if args.embedding_batch_size < 1 or args.embedding_samples_per_epoch < 1:
        parser.error(
            "--embedding-batch-size and --embedding-samples-per-epoch must be positive."
        )
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    models = []
    for family in MODEL_CLASSES:
        run_id = getattr(args, f"{family}_run_id")
        if run_id:
            models.append(
                _load_model(
                    client,
                    family,
                    run_id,
                    getattr(args, f"{family}_artifact"),
                    device,
                )
            )

    selected_by_model = []
    for loaded in models:
        dataloader = MRI_Dataloader()
        selected_by_model.append(_selected_trajectories(dataloader, loaded.params))
    common_ids = set(selected_by_model[0])
    for selected in selected_by_model[1:]:
        common_ids.intersection_update(selected)

    validation = _validation_samples()
    with open(DatasetPaths().patient_to_idx(), encoding="utf-8") as file:
        patient_to_idx = json.load(file)
    common_ids.intersection_update(validation)
    if not common_ids:
        raise ValueError(
            "The supplied runs have no common validation lesions after "
            "applying each run's scan-count and growth filters."
        )

    trajectories = {
        lesion_id: selected_by_model[0][lesion_id]
        for lesion_id in sorted(common_ids)
    }
    cases = _make_cases(trajectories, validation, args.split)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError(f"No {args.split} validation cases were selected.")

    print(
        f"Evaluating {len(models)} model(s), {len(trajectories)} shared lesion(s), "
        f"{len(cases)} target(s), {args.resolution}^3 = "
        f"{args.resolution**3:,} query points per target on {device}."
    )

    lstm_datasets = {
        loaded.run_id: _lstm_dataset(list(trajectories.values()), loaded)
        for loaded in models
        if loaded.family == "lstm"
    }
    trajectory_indices = {
        lesion_id: index for index, lesion_id in enumerate(trajectories)
    }

    rows = []
    embedding_trajectory = None
    embedding_model_idx: int | None = None
    embedding_trajectory_dates: list[str] = []
    embedding_lstm_datasets = {}
    if not args.skip_embedding_test:
        embedding_dataloader = MRI_Dataloader()
        embedding_trajectory = embedding_dataloader.find_lesion_trajectory(
            *EMBEDDING_TRAJECTORY
        )
        if embedding_trajectory is None:
            raise FileNotFoundError(
                "Could not load embedding trajectory "
                f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]}."
            )
        embedding_trajectory_dates = [
            str(date)
            for date in (
                embedding_trajectory.allowed_dates
                if hasattr(embedding_trajectory, "allowed_dates")
                else embedding_trajectory.dates
            )
        ]
        if not embedding_trajectory_dates:
            raise ValueError(
                "Embedding trajectory "
                f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]} "
                "has no timepoints."
            )
        embedding_patient_idx = patient_to_idx.get(
            embedding_trajectory.patient_id
        )
        if isinstance(embedding_patient_idx, str):
            embedding_patient_idx = int(embedding_patient_idx)
        elif not isinstance(embedding_patient_idx, int):
            raise ValueError(
                "Embedding trajectory patient "
                f"{embedding_trajectory.patient_id} is missing from patient_to_idx."
            )
        embedding_model_idx = embedding_patient_idx * 100 + EMBEDDING_TRAJECTORY[1]
        embedding_lstm_datasets = {
            loaded.run_id: _lstm_dataset([embedding_trajectory], loaded)
            for loaded in models
            if loaded.family == "lstm"
        }
    mlflow.set_experiment("Validation")
    mlflow.start_run(run_name="Uniform Grid Validation")
    validation_params: dict[str, str] = {
        "resolution_per_axis": str(args.resolution),
        "query_points_per_target": str(args.resolution**3),
        "decision_threshold": "0.5",
        "split": args.split,
        "embedding_test_enabled": str(not args.skip_embedding_test),
    }
    if not args.skip_embedding_test:
        validation_params.update(
            {
                "embedding_adaptation_epochs": str(args.embedding_epochs),
                "embedding_combination_phase_epochs": str(
                    min(args.embedding_combination_epochs, args.embedding_epochs)
                ),
                "embedding_initial_learning_rate": str(args.embedding_lr),
                "embedding_direct_phase_learning_rate": str(
                    args.embedding_lr * 0.01
                ),
                "embedding_batch_size": str(args.embedding_batch_size),
                "embedding_samples_per_epoch": str(
                    args.embedding_samples_per_epoch
                ),
                "embedding_trajectory": (
                    f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]}"
                ),
            }
        )
    for loaded in models:
        validation_params[f"{loaded.family}_run_id"] = loaded.run_id
    mlflow.log_params(validation_params)
    logged_params = set(validation_params)

    def record_result(
        loaded: LoadedModel,
        model_name: str,
        case: EvaluationCase,
        result_sums: dict[str, float],
    ) -> None:
        soft_dice, hard_dice = _dice_results(result_sums)
        voxel_volume_cm3 = (200.0 / (args.resolution - 1)) ** 3 / 1000.0
        predicted_volume_cm3 = (
            result_sums["hard_prediction"] * voxel_volume_cm3
        )
        target_volume_cm3 = result_sums["target"] * voxel_volume_cm3
        rows.append(
            {
                "model": model_name,
                "run_id": loaded.run_id,
                "lesion_id": case.lesion_id,
                "split": case.split,
                "target_date": case.target_date,
                "days_from_endpoint": case.days_from_endpoint,
                "resolution_per_axis": args.resolution,
                "soft_dice": soft_dice,
                "hard_dice_at_0.5": hard_dice,
                "wrongly_predicted_volume_cm3": abs(
                    predicted_volume_cm3 - target_volume_cm3
                ),
            }
        )
    for loaded in models:
        for case in _print_progress(
            cases,
            f"Evaluating {loaded.family} ({loaded.run_id[:8]})",
            "targets",
        ):
            target_volume = _load_target_volume(case)
            if loaded.family == "lstm":
                sums = _infer_lstm(
                    loaded,
                    case,
                    target_volume,
                    trajectory_indices[case.lesion_id],
                    lstm_datasets[loaded.run_id],
                    args.resolution,
                    args.chunk_size,
                    device,
                )
            else:
                sums = _infer_continuous_model(
                    loaded,
                    target_volume,
                    int(patient_to_idx[case.trajectory.patient_id]) * 100
                    + int(case.trajectory.label_id),
                    args.resolution,
                    args.chunk_size,
                    device,
                )

            record_result(loaded, loaded.family, case, sums)

        model_metrics = _summary_metrics(rows, loaded.family)
        if model_metrics:
            mlflow.log_metrics(model_metrics)

        if args.skip_embedding_test:
            continue

        if embedding_trajectory is None or embedding_model_idx is None:
            raise RuntimeError("Embedding trajectory setup was not completed.")
        print(
            f"Fitting frozen-network embedding for {loaded.family} on "
            f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]} using "
            f"{len(embedding_trajectory_dates)} timepoint(s) and "
            f"{args.embedding_samples_per_epoch} sampled example(s) per epoch."
        )
        embedding_case = EvaluationCase(
            f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]}",
            embedding_trajectory,
            "trajectory",
            embedding_trajectory_dates[-1],
            None,
            tuple(embedding_trajectory_dates),
        )
        fitted_embedding = _fit_embedding(
            loaded,
            embedding_case,
            0,
            embedding_lstm_datasets.get(loaded.run_id),
            args.embedding_epochs,
            args.embedding_combination_epochs,
            args.embedding_lr,
            args.embedding_batch_size,
            args.embedding_samples_per_epoch,
            device,
        )

        for date_index, date in enumerate(
            _print_progress(
                embedding_trajectory_dates,
                f"Evaluating {loaded.family} fitted trajectory",
                "timepoints",
            )
        ):
            trajectory_case = EvaluationCase(
                embedding_case.lesion_id,
                embedding_trajectory,
                "trajectory",
                date,
                None,
                tuple(embedding_trajectory_dates[:date_index])
                or (date,),
            )
            target_volume = _load_target_volume(trajectory_case)
            if loaded.family == "lstm":
                adapted_sums = _infer_lstm(
                    loaded,
                    trajectory_case,
                    target_volume,
                    0,
                    embedding_lstm_datasets[loaded.run_id],
                    args.resolution,
                    args.chunk_size,
                    device,
                    embedding_override=fitted_embedding,
                )
            else:
                adapted_sums = _infer_continuous_model(
                    loaded,
                    target_volume,
                    embedding_model_idx,
                    args.resolution,
                    args.chunk_size,
                    device,
                    embedding_override=fitted_embedding,
                )
            record_result(
                loaded, f"{loaded.family}_embedding", trajectory_case, adapted_sums
            )

        embedding_metrics = _summary_metrics(rows, f"{loaded.family}_embedding")
        if embedding_metrics:
            mlflow.log_metrics(embedding_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-target results written to {args.output}")
    for loaded in models:
        model_names = [loaded.family]
        if not args.skip_embedding_test:
            model_names.append(f"{loaded.family}_embedding")
        for model_name in model_names:
            splits = (
                ("trajectory",)
                if model_name.endswith("_embedding")
                else ("interpolation", "extrapolation")
            )
            for split in splits:
                split_rows = [
                    row
                    for row in rows
                    if row["model"] == model_name and row["split"] == split
                ]
                if split_rows:
                    prefix = f"{model_name}_{split}"
                    dice_loss = float(
                        np.mean([1.0 - row["soft_dice"] for row in split_rows])
                    )
                    volume_errors = [
                        row["wrongly_predicted_volume_cm3"] for row in split_rows
                    ]
                    validation_params.update(
                        {
                            f"{prefix}_dice_loss": str(dice_loss),
                            f"{prefix}_dice_score": str(1.0 - dice_loss),
                            f"{prefix}_median_wrongly_predicted_volume_cm3": str(
                                float(np.median(volume_errors))
                            ),
                            f"{prefix}_average_wrongly_predicted_volume_cm3": str(
                                float(np.mean(volume_errors))
                            ),
                            f"{prefix}_wrongly_predicted_volume_cm3": str(
                                float(np.sum(volume_errors))
                            ),
                            f"{prefix}_hard_dice_score": str(
                                float(
                                    np.mean(
                                        [
                                            row["hard_dice_at_0.5"]
                                            for row in split_rows
                                        ]
                                    )
                                )
                            ),
                        }
                    )
                    print(
                        f"{model_name} {split}: "
                        f"dice loss={dice_loss:.5f}, "
                        f"dice score={1.0 - dice_loss:.5f}, "
                        f"median volume error={np.median(volume_errors):.5f} cm^3, "
                        f"average volume error={np.mean(volume_errors):.5f} cm^3, "
                        f"total volume error={np.sum(volume_errors):.5f} cm^3, "
                        f"hard Dice={np.mean([row['hard_dice_at_0.5'] for row in split_rows]):.5f} "
                        f"(n={len(split_rows)})"
                    )
    summary_params = {
        key: value
        for key, value in validation_params.items()
        if key not in logged_params
    }
    if summary_params:
        mlflow.log_params(summary_params)
    mlflow.log_artifact(str(args.output))
    print(
        "Logged validation parameters and per-target CSV to MLflow "
        f"experiment 'Validation' (run {mlflow.active_run().info.run_id})."
    )
    mlflow.end_run()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        raise