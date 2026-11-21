"""Compare saved MLflow INR, Neural ODE, and LSTM models with dilation baselines.

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

Unless --skip-embedding-test is set, INR, Neural ODE, and LSTM models also fit
one embedding on YG_WCONBI42RLEZ lesion 1 and report fitted-embedding Dice for
every timepoint. Global-rate and lesion-specific-rate dilation baselines are
computed only from non-validation scans and also evaluated on the embedding
trajectory.

Use --plot-contact-sheets to save contact sheets for up to five validation
lesions and each model's fitted-embedding trajectory.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
import zlib
from collections.abc import Callable
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
from scipy.ndimage import binary_dilation, map_coordinates

from utils.dataloader import LesionDataset
from utils.dataloader_lstm import LesionSequenceDataset
from utils.loss_bce_dice import Loss_BCE_Dice
from utils.model_inr import LesionINR
from utils.model_lstm import LesionLSTM, LesionLatentLSTM
from utils.model_dialation_patient_specific import calculate_lesion_growth_rate
from utils import model_neural_ode_validation
from utils import model_neural_ode_translation_validation
from utils.model_point_lstm import PointLSTM
from utils.dataloader_point_lstm import (
    ScanCache,
    batch_to_device,
    collate_history,
    history_arrays,
)
from utils.mri_dataloader import MRI_Dataloader
from utils.paths import DatasetPaths


MODEL_CLASSES = {
    "inr": (LesionINR,),
    "node": (
        model_neural_ode_validation.NeuralODE_INR,
        model_neural_ode_translation_validation.NeuralODETranslationINR,
    ),
    "lstm": (LesionLSTM, LesionLatentLSTM),
    "plstm": (PointLSTM,),
}
EMBEDDING_SUPPORTED_FAMILIES = {"inr", "node", "lstm"}
DILATION_BASELINE_NAMES = ("dilation_global", "dilation_lesion_specific")
EMBEDDING_TRAJECTORY = ("YG_WCONBI42RLEZ", 1)
VALIDATION_CACHE_DIR = (
    Path(__file__).resolve().parent / "cache" / "validation_arena"
)
CACHE_VERSION = 2


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
    extrapolation_timepoints: int | None = None


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
            if target_index != -1:
                if not 0 <= target_index < len(dates):
                    raise ValueError(
                        f"Invalid interpolation index for lesion {lesion_id}."
                    )
                target_date = str(dates[target_index])
                history = tuple(str(date) for date in dates[:target_index])
                if not history:
                    history = (target_date,)
                cases.append(
                    EvaluationCase(
                        lesion_id,
                        trajectory,
                        "interpolation",
                        target_date,
                        None,
                        history,
                    )
                )
        if split in ("both", "extrapolation"):
            held_out_list = [str(date) for date in sample["extrapolation_dates"]]
            held_out_dates = set(held_out_list)
            held_out_steps = sample["extrapolation_T_minus"]
            interpolation_index = int(sample["interpolation"])
            # -1 means the lesion has no interpolation scan.
            interpolation_date = (
                str(dates[interpolation_index]) if interpolation_index != -1 else None
            )
            if interpolation_date is not None and interpolation_date in held_out_dates:
                raise ValueError(
                    f"{lesion_id}: interpolation date {interpolation_date} is also "
                    "an extrapolation date."
                )
            history = tuple(
                str(date)
                for date in dates
                if str(date) not in held_out_dates
                and str(date) != interpolation_date
            )
            for step, date in zip(held_out_steps, held_out_list):
                cases.append(
                    EvaluationCase(
                        lesion_id,
                        trajectory,
                        "extrapolation",
                        str(date),
                        float(step),
                        history,
                        len(held_out_dates),
                    )
                )
    return cases


def _dilation_growth_rates(
    trajectories: dict[str, Any],
    validation_samples: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, float]]:
    """Estimate rates from training dates only, excluding validation targets."""
    lesion_rates = {}
    for lesion_id, trajectory in trajectories.items():
        training_dates = _training_dates_for_rate(trajectory, validation_samples)
        lesion_rates[lesion_id] = float(
            calculate_lesion_growth_rate(trajectory, training_dates)
        )
    if not lesion_rates:
        raise ValueError("Cannot calculate dilation growth rates without trajectories.")
    return float(np.mean(list(lesion_rates.values()))), lesion_rates


def _training_dates_for_rate(
    trajectory: Any,
    validation_samples: dict[str, dict[str, Any]],
) -> list[str]:
    dates = [
        str(date)
        for date in (
            trajectory.allowed_dates
            if hasattr(trajectory, "allowed_dates")
            else trajectory.dates
        )
    ]
    lesion_id = f"{trajectory.patient_id}_{trajectory.label_id}"
    sample = validation_samples.get(lesion_id)
    if sample is None:
        training_dates = dates
    else:
        held_out_dates = set(str(date) for date in sample["extrapolation_dates"])
        interpolation_index = int(sample["interpolation"])
        if interpolation_index != -1:
            if not 0 <= interpolation_index < len(dates):
                raise ValueError(f"Invalid interpolation index for lesion {lesion_id}.")
            held_out_dates.add(str(dates[interpolation_index]))
        training_dates = [date for date in dates if date not in held_out_dates]
    if len(training_dates) < 2:
        raise ValueError(
            f"Cannot estimate a dilation growth rate for {lesion_id}: "
            f"only {len(training_dates)} training scan(s) remain after "
            "excluding validation timepoints."
        )
    return training_dates


def _seed_embedding_fit(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    sums["target"] += float(targets.sum())
    predicted = probabilities >= 0.5
    sums["hard_prediction"] += int(predicted.sum())
    sums["hard_intersection"] += int(np.logical_and(predicted, targets).sum())


def _dice_results(sums: dict[str, float]) -> float:
    return (
        2.0 * sums["hard_intersection"] + 1e-6
    ) / (sums["hard_prediction"] + sums["target"] + 1e-6)


def _cache_run_component(run_id: str) -> str:
    component = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in run_id
    )
    return component or "unknown-run"


def _prediction_cache_path(
    run_id: str,
    model_name: str,
    case: EvaluationCase,
    resolution: int,
    chunk_size: int,
    device: torch.device,
    growth_rate: float | None = None,
    inference_key: str | None = None,
) -> Path:
    cache_key = json.dumps(
        {
            "version": CACHE_VERSION,
            "run_id": run_id,
            "model": model_name,
            "lesion_id": case.lesion_id,
            "split": case.split,
            "target_date": case.target_date,
            "days_from_endpoint": case.days_from_endpoint,
            "history_dates": case.history_dates,
            "resolution": resolution,
            "chunk_size": chunk_size,
            "device": str(device),
            "growth_rate": growth_rate,
            "inference_key": inference_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    run_dir = VALIDATION_CACHE_DIR / _cache_run_component(run_id)
    return run_dir / f"{digest}.npz"


def _load_cached_masks(
    path: Path, point_count: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cache:
        cached_point_count = int(cache["point_count"])
        if cached_point_count != point_count:
            raise ValueError(
                f"Cached evaluation at {path} has {cached_point_count} points; "
                f"expected {point_count}."
            )
        expected_bytes = (point_count + 7) // 8
        predictions = np.asarray(cache["predictions"], dtype=np.uint8)
        targets = np.asarray(cache["targets"], dtype=np.uint8)
        if predictions.size != expected_bytes or targets.size != expected_bytes:
            raise ValueError(f"Cached evaluation at {path} has invalid mask data.")
        return (
            np.unpackbits(predictions, count=point_count).astype(bool, copy=False),
            np.unpackbits(targets, count=point_count).astype(bool, copy=False),
        )


def _save_cached_masks(
    path: Path,
    point_count: int,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    predictions = np.asarray(predictions, dtype=bool).reshape(-1)
    targets = np.asarray(targets, dtype=bool).reshape(-1)
    if predictions.size != point_count or targets.size != point_count:
        raise ValueError(
            "Evaluation masks do not match the requested number of query points."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, suffix=".npz", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                point_count=np.int64(point_count),
                predictions=np.packbits(predictions),
                targets=np.packbits(targets),
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _cached_case_sums(
    path: Path,
    case: EvaluationCase,
    point_count: int,
    infer: Callable[
        [TargetVolume], tuple[np.ndarray, np.ndarray]
    ],
) -> dict[str, float]:
    masks = _load_cached_masks(path, point_count)
    if masks is None:
        target_volume = _load_target_volume(case)
        masks = infer(target_volume)
        _save_cached_masks(path, point_count, *masks)

    sums: dict[str, float] = {
        "target": 0.0,
        "hard_prediction": 0.0,
        "hard_intersection": 0.0,
    }
    _dice_sums(sums, *masks)
    return sums


def _embedding_cache_path(
    loaded: LoadedModel,
    args: argparse.Namespace,
    trajectory_dates: list[str],
    device: torch.device,
) -> Path:
    cache_key = json.dumps(
        {
            "version": CACHE_VERSION,
            "run_id": loaded.run_id,
            "model": loaded.family,
            "epochs": args.embedding_epochs,
            "combination_epochs": args.embedding_combination_epochs,
            "learning_rate": args.embedding_lr,
            "batch_size": args.embedding_batch_size,
            "samples_per_epoch": args.embedding_samples_per_epoch,
            "seed": args.embedding_seed,
            "trajectory": EMBEDDING_TRAJECTORY,
            "trajectory_dates": trajectory_dates,
            "device": str(device),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    run_dir = VALIDATION_CACHE_DIR / _cache_run_component(loaded.run_id)
    return run_dir / f"embedding-{digest}.npy"


def _load_or_fit_embedding(
    path: Path,
    device: torch.device,
    fit: Callable[[], torch.Tensor],
) -> torch.Tensor:
    if path.is_file():
        embedding = np.load(path, allow_pickle=False)
        if embedding.ndim != 1 or not np.isfinite(embedding).all():
            raise ValueError(f"Cached embedding at {path} is invalid.")
        return torch.from_numpy(embedding).to(device)

    fitted_embedding = fit()
    embedding_array = (
        fitted_embedding.detach().to(device="cpu", dtype=torch.float32).numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, suffix=".npy", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, embedding_array, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return fitted_embedding


def _embedding_fingerprint(embedding: torch.Tensor) -> str:
    embedding_array = (
        embedding.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    )
    return hashlib.sha256(embedding_array.tobytes()).hexdigest()


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
        dice_score = _average_metric(split_rows, "dice_score")
        volume_errors = [
            row["wrongly_predicted_volume_cm3"] for row in split_rows
        ]
        metrics.update(
            {
                f"{prefix}_dice_loss": 1.0 - dice_score,
                f"{prefix}_dice_score": dice_score,
                f"{prefix}_median_wrongly_predicted_volume_cm3": float(
                    np.median(volume_errors)
                ),
                f"{prefix}_average_wrongly_predicted_volume_cm3": float(
                    _average_metric(split_rows, "wrongly_predicted_volume_cm3")
                ),
                f"{prefix}_wrongly_predicted_volume_cm3": float(
                    np.sum(volume_errors)
                ),
            }
        )
    return metrics


def _extrapolation_weights(rows: list[dict[str, Any]]) -> dict[str, float]:
    timepoints_by_lesion: dict[str, int] = {}
    for row in rows:
        lesion_id = row["lesion_id"]
        timepoint_count = row["extrapolation_timepoints"]
        if timepoint_count is None or timepoint_count < 1:
            raise ValueError(
                f"Invalid extrapolation timepoint count for lesion {lesion_id}."
            )
        previous_count = timepoints_by_lesion.setdefault(
            lesion_id, int(timepoint_count)
        )
        if previous_count != timepoint_count:
            raise ValueError(
                f"Inconsistent extrapolation timepoint count for lesion {lesion_id}."
            )
    max_timepoints = max(timepoints_by_lesion.values())
    return {
        lesion_id: timepoint_count / max_timepoints
        for lesion_id, timepoint_count in timepoints_by_lesion.items()
    }


def _average_metric(rows: list[dict[str, Any]], metric: str) -> float:
    if rows[0]["split"] != "extrapolation":
        return float(np.mean([row[metric] for row in rows]))

    values_by_lesion: dict[str, list[float]] = {}
    for row in rows:
        values_by_lesion.setdefault(row["lesion_id"], []).append(row[metric])
    weights = _extrapolation_weights(rows)
    lesion_averages = [
        float(np.mean(values)) for values in values_by_lesion.values()
    ]
    lesion_weights = [weights[lesion_id] for lesion_id in values_by_lesion]
    return float(np.average(lesion_averages, weights=lesion_weights))


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
) -> tuple[np.ndarray, np.ndarray]:
    count = resolution**3
    predictions = np.empty(count, dtype=bool)
    target_mask = np.empty(count, dtype=bool)
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
                coords, chunk_targets = _coordinates_and_labels(
                    target_volume, start, end, resolution
                )
                coords_tensor = torch.from_numpy(coords).unsqueeze(0).to(device)
                logits = loaded.model(coords_tensor, patient_tensor).reshape(-1)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions[start:end] = probabilities >= 0.5
                target_mask[start:end] = chunk_targets
    finally:
        if hook is not None:
            hook.remove()
    return predictions, target_mask


def _infer_dilation_baseline(
    case: EvaluationCase,
    target_volume: TargetVolume,
    growth_rate: float,
    resolution: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = case.trajectory
    dates = list(
        trajectory.allowed_dates
        if hasattr(trajectory, "allowed_dates")
        else trajectory.dates
    )
    history_dates = case.history_dates or tuple(str(date) for date in dates)
    (base_labels, base_time), affine = trajectory.load_labels_for_inr(
        selected_date=str(history_dates[-1]),
        absolute_day_number=True,
        affine=True,
    )
    base_mask = np.asarray(base_labels) > 0.5
    elapsed_days = max(0.0, target_volume.time_point - float(base_time))
    iterations = int(np.round(elapsed_days * growth_rate))
    predicted_mask = (
        binary_dilation(base_mask, iterations=iterations)
        if iterations > 0
        else base_mask
    )

    volume_center_voxel = (np.asarray(base_mask.shape, dtype=np.float64) - 1) / 2
    volume_center_phys = affine[:3, :3] @ volume_center_voxel + affine[:3, 3]
    inverse_affine = np.linalg.inv(affine)
    count = resolution**3
    predictions = np.empty(count, dtype=bool)
    targets_array = np.empty(count, dtype=bool)

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        coords, targets = _coordinates_and_labels(
            target_volume, start, end, resolution
        )
        physical_coords = (
            volume_center_phys + coords[:, :3].astype(np.float64) * 100.0
        )
        physical_homogeneous = np.concatenate(
            (physical_coords, np.ones((len(coords), 1))), axis=1
        )
        voxel_coords = physical_homogeneous @ inverse_affine.T
        pixel_coords = np.rint(voxel_coords[:, :3]).astype(np.int64)
        pixel_coords = np.clip(
            pixel_coords, 0, np.asarray(base_mask.shape, dtype=np.int64) - 1
        )
        sampled_mask = predicted_mask[
            pixel_coords[:, 0], pixel_coords[:, 1], pixel_coords[:, 2]
        ]
        predictions[start:end] = sampled_mask
        targets_array[start:end] = targets

    return predictions, targets_array


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


def _plot_lstm_contact_sheet(
    model: torch.nn.Module,
    trajectory: Any,
    sequence_dataset: LesionSequenceDataset,
    trajectory_index: int,
    output_path: Path,
    device: torch.device,
    embedding_override: torch.Tensor | None = None,
) -> None:
    from matplotlib import pyplot as plt
    from utils.train_plotting_lstm import plot_contanct_sheet

    dates = list(
        trajectory.allowed_dates
        if hasattr(trajectory, "allowed_dates")
        else trajectory.dates
    )
    if not dates:
        raise ValueError(
            f"Cannot plot LSTM contact sheet for "
            f"{trajectory.patient_id}_{trajectory.label_id}: no dates."
        )
    resampled = sequence_dataset._compute_resampled_trajectory(
        trajectory_index, dates
    )
    trajectory.grids = np.stack(
        [resampled["grids"][date] for date in dates], axis=0
    )[:, None]
    trajectory.times = np.asarray(
        [resampled["times"][date] for date in dates], dtype=np.float32
    )
    trajectory.embedding_id = sequence_dataset.get_embedding_id(trajectory)
    trajectory.canonical_center = resampled["center"]
    trajectory.canonical_extent_mm = float(resampled["extent"])

    plot_config = argparse.Namespace(
        device=device,
        time_evolution_steps=12,
        lstm_grid_size=tuple(sequence_dataset.grid_size),
        full_size_side_length=500,
        time_evolution_side_length=100,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hook = (
        _embedding_override(model, embedding_override)
        if embedding_override is not None
        else None
    )
    figure = None
    try:
        figure = plot_contanct_sheet(
            model,
            epoch=0,
            trj=trajectory,
            config=plot_config,
            final_side_length=True,
            jupyter_mode=False,
            output_path=str(output_path),
        )
    finally:
        if hook is not None:
            hook.remove()
        if figure is not None:
            plt.close(figure)


def _plot_continuous_contact_sheet(
    model: torch.nn.Module,
    trajectory: Any,
    embedding_id: int,
    output_path: Path,
    device: torch.device,
    embedding_override: torch.Tensor | None = None,
) -> None:
    from matplotlib import pyplot as plt
    from utils.train_plotting import plot_contanct_sheet

    trajectory.embedding_id = embedding_id
    plot_config = argparse.Namespace(
        device=device,
        time_evolution_steps=12,
        full_size_side_length=500,
        time_evolution_side_length=128,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hook = (
        _embedding_override(model, embedding_override)
        if embedding_override is not None
        else None
    )
    figure = None
    try:
        figure = plot_contanct_sheet(
            model,
            epoch=0,
            trj=trajectory,
            config=plot_config,
            final_side_length=False,
            jupyter_mode=False,
            output_path=str(output_path),
        )
    finally:
        if hook is not None:
            hook.remove()
        if figure is not None:
            plt.close(figure)


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
) -> tuple[np.ndarray, np.ndarray]:
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
    count = resolution**3
    predictions = np.empty(count, dtype=bool)
    targets_array = np.empty(count, dtype=bool)
    for start in _print_progress(
        range(0, count, chunk_size),
        f"LSTM {case.target_date} grid",
        "chunks",
    ):
        end = min(start + chunk_size, count)
        coords, targets = _coordinates_and_labels(
            target_volume, start, end, resolution
        )
        physical_coords = (
            target_volume.center_phys
            + coords[:, :3].astype(np.float64) * 100.0
        )
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
        predictions[start:end] = probabilities >= 0.5
        targets_array[start:end] = targets
    return predictions, targets_array


def _infer_point_lstm(
    loaded: LoadedModel,
    case: EvaluationCase,
    target_volume: TargetVolume,
    patient_idx: int,
    scan_cache: ScanCache,
    resolution: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = loaded.model
    history_dates = [
        date for date in case.history_dates if date != case.target_date
    ][-model.max_history_len:]
    if not history_dates:
        raise ValueError(
            f"No observed history before {case.target_date} for {case.lesion_id}."
        )

    dataset_params = _literal_param(loaded.params, "dataset_params", {})
    ring_mm = (
        float(dataset_params.get("ring_mm", 3.0))
        if isinstance(dataset_params, dict)
        else 3.0
    )
    entries = [scan_cache.get(case.trajectory, date) for date in history_dates]
    rng = np.random.default_rng(
        zlib.crc32(f"{case.lesion_id}|{case.target_date}".encode())
    )
    history = batch_to_device(
        collate_history([history_arrays(entries, model.history_points, rng, ring_mm)]),
        device,
    )
    patient_tensor = torch.tensor([patient_idx], dtype=torch.long, device=device)
    target_time = torch.tensor(
        [target_volume.time_point], dtype=torch.float32, device=device
    )

    count = resolution**3
    predictions = np.empty(count, dtype=bool)
    targets_array = np.empty(count, dtype=bool)
    with torch.inference_mode():
        z = model.latent(model.encode_history(history), target_time, patient_tensor)
        for start in _print_progress(
            range(0, count, chunk_size),
            f"plstm {target_volume.time_point:g}d grid",
            "chunks",
        ):
            end = min(start + chunk_size, count)
            coords, targets = _coordinates_and_labels(
                target_volume, start, end, resolution
            )
            sdf = None
            if model.use_sdf_prior:
                sdf = torch.from_numpy(entries[-1].sdf_at_model(coords[:, :3]))
                sdf = sdf.view(1, -1, 1).to(device)
            coords_tensor = torch.from_numpy(coords).unsqueeze(0).to(device)
            logits = model.decode(coords_tensor, z, sdf).reshape(-1)
            predictions[start:end] = torch.sigmoid(logits).cpu().numpy() >= 0.5
            targets_array[start:end] = targets
    return predictions, targets_array


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
                    epoch_samples = _prepare_continuous_support(
                        loaded, case, samples_per_epoch
                    )
                    shuffled = random.sample(epoch_samples, len(epoch_samples))
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
            "Evaluate MLflow INR, Neural ODE, and/or LSTM runs alongside "
            "calculated dilation baselines on a shared uniform spatial grid."
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
        "--plot-contact-sheets",
        "--plot-lstm-contact-sheets",
        dest="plot_contact_sheets",
        action="store_true",
        help=(
            "Save contact sheets for up to five distinct validation lesions "
            "per model and, unless embedding testing is skipped, each fitted "
            "embedding trajectory. The LSTM-specific option name remains an "
            "alias."
        ),
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
        "--embedding-seed",
        type=int,
        default=42,
        help="Random seed reset before each embedding fit (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().with_name("validation_arena_results.csv"),
        help=(
            "Per-target results CSV path (default: validation_arena_results.csv "
            "beside this script)"
        ),
    )
    args = parser.parse_args()
    if not any(getattr(args, f"{family}_run_id") for family in MODEL_CLASSES):
        parser.error(
            "Provide at least one of --inr-run-id, --node-run-id, "
            "--lstm-run-id, or --plstm-run-id."
        )
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
    global_growth_rate, lesion_growth_rates = _dilation_growth_rates(
        trajectories, validation
    )

    print(
        f"Evaluating {len(models)} model(s) and 2 dilation baselines, "
        f"{len(trajectories)} shared lesion(s), "
        f"{len(cases)} target(s), {args.resolution}^3 = "
        f"{args.resolution**3:,} query points per target on {device}."
    )
    print(
        "Calculated global dilation growth rate: "
        f"{global_growth_rate:.6f} voxels/day."
    )

    lstm_datasets = {
        loaded.run_id: _lstm_dataset(list(trajectories.values()), loaded)
        for loaded in models
        if loaded.family == "lstm"
    }
    trajectory_indices = {
        lesion_id: index for index, lesion_id in enumerate(trajectories)
    }
    point_lstm_caches = {
        loaded.run_id: ScanCache(sdf_clip_mm=loaded.model.sdf_clip_mm)
        for loaded in models
        if loaded.family == "plstm"
    }

    rows = []
    embedding_families = {
        loaded.family
        for loaded in models
        if loaded.family in EMBEDDING_SUPPORTED_FAMILIES
    }
    embedding_trajectory = None
    embedding_model_idx: int | None = None
    embedding_trajectory_dates: list[str] = []
    embedding_dilation_growth_rate: float | None = None
    embedding_lstm_datasets = {}
    if not args.skip_embedding_test and embedding_families:
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
        embedding_dilation_growth_rate = float(
            calculate_lesion_growth_rate(
                embedding_trajectory,
                _training_dates_for_rate(embedding_trajectory, validation),
            )
        )
        embedding_lstm_datasets = {
            loaded.run_id: _lstm_dataset([embedding_trajectory], loaded)
            for loaded in models
            if loaded.family == "lstm"
        }
    mlflow.set_experiment("Uniform Grid Validation")
    mlflow.start_run(run_name="Uniform Grid Validation")
    validation_params: dict[str, str] = {
        "resolution_per_axis": str(args.resolution),
        "query_points_per_target": str(args.resolution**3),
        "decision_threshold": "0.5",
        "split": args.split,
        "contact_sheets_enabled": str(args.plot_contact_sheets),
        "embedding_test_enabled": str(
            not args.skip_embedding_test and bool(embedding_families)
        ),
        "dilation_global_growth_rate_voxels_per_day": str(global_growth_rate),
        "dilation_lesion_growth_rate_count": str(len(lesion_growth_rates)),
        "dilation_growth_rate_excludes_validation_timepoints": "true",
        "inr_run_id": args.inr_run_id or "",
        "node_run_id": args.node_run_id or "",
        "lstm_run_id": args.lstm_run_id or "",
        "plstm_run_id": args.plstm_run_id or "",
    }
    if embedding_dilation_growth_rate is not None:
        validation_params[
            "dilation_embedding_lesion_growth_rate_voxels_per_day"
        ] = str(embedding_dilation_growth_rate)
    for loaded in models:
        validation_params[f"{loaded.family}_embedding_test_enabled"] = str(
            not args.skip_embedding_test
            and loaded.family in EMBEDDING_SUPPORTED_FAMILIES
        )
    for baseline_name in DILATION_BASELINE_NAMES:
        validation_params[f"{baseline_name}_embedding_test_enabled"] = str(
            not args.skip_embedding_test and embedding_trajectory is not None
        )
    if not args.skip_embedding_test and embedding_families:
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
                "embedding_seed": str(args.embedding_seed),
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
        model_name: str,
        run_id: str,
        case: EvaluationCase,
        result_sums: dict[str, float],
        growth_rate: float | None = None,
    ) -> None:
        dice_score = _dice_results(result_sums)
        voxel_volume_cm3 = (200.0 / (args.resolution - 1)) ** 3 / 1000.0
        predicted_volume_cm3 = (
            result_sums["hard_prediction"] * voxel_volume_cm3
        )
        target_volume_cm3 = result_sums["target"] * voxel_volume_cm3
        rows.append(
            {
                "model": model_name,
                "run_id": run_id,
                "lesion_id": case.lesion_id,
                "split": case.split,
                "target_date": case.target_date,
                "days_from_endpoint": case.days_from_endpoint,
                "extrapolation_timepoints": case.extrapolation_timepoints,
                "resolution_per_axis": args.resolution,
                "growth_rate_voxels_per_day": growth_rate,
                "dice_score": dice_score,
                "wrongly_predicted_volume_cm3": abs(
                    predicted_volume_cm3 - target_volume_cm3
                ),
            }
        )

    def cached_result(
        model_name: str,
        run_id: str,
        case: EvaluationCase,
        infer: Callable[[TargetVolume], tuple[np.ndarray, np.ndarray]],
        growth_rate: float | None = None,
        inference_key: str | None = None,
    ) -> dict[str, float]:
        cache_path = _prediction_cache_path(
            run_id,
            model_name,
            case,
            args.resolution,
            args.chunk_size,
            device,
            growth_rate=growth_rate,
            inference_key=inference_key,
        )
        return _cached_case_sums(
            cache_path, case, args.resolution**3, infer
        )

    for loaded in models:
        for case in _print_progress(
            cases,
            f"Evaluating {loaded.family} ({loaded.run_id[:8]})",
            "targets",
        ):
            if loaded.family == "lstm":
                infer = lambda target_volume: _infer_lstm(
                    loaded,
                    case,
                    target_volume,
                    trajectory_indices[case.lesion_id],
                    lstm_datasets[loaded.run_id],
                    args.resolution,
                    args.chunk_size,
                    device,
                )
            elif loaded.family == "plstm":
                patient_idx = (
                    int(patient_to_idx[case.trajectory.patient_id]) * 100
                    + int(case.trajectory.label_id)
                )
                infer = lambda target_volume: _infer_point_lstm(
                    loaded,
                    case,
                    target_volume,
                    patient_idx,
                    point_lstm_caches[loaded.run_id],
                    args.resolution,
                    args.chunk_size,
                    device,
                )
            else:
                patient_idx = (
                    int(patient_to_idx[case.trajectory.patient_id]) * 100
                    + int(case.trajectory.label_id)
                )
                infer = lambda target_volume: _infer_continuous_model(
                    loaded,
                    target_volume,
                    patient_idx,
                    args.resolution,
                    args.chunk_size,
                    device,
                )
            sums = cached_result(
                loaded.family, loaded.run_id, case, infer
            )

            record_result(loaded.family, loaded.run_id, case, sums)

        if args.plot_contact_sheets:
            plotted_lesions = set()
            normal_plot_count = 0
            contact_sheet_dir = args.output.with_name(
                f"{args.output.stem}_contact_sheets"
            )
            for case in cases:
                if case.lesion_id in plotted_lesions:
                    continue
                plotted_lesions.add(case.lesion_id)
                normal_plot_count += 1
                output_path = contact_sheet_dir / (
                    f"{loaded.family}_normal_{normal_plot_count:02d}_"
                    f"{case.lesion_id}.png"
                )
                if loaded.family == "lstm":
                    _plot_lstm_contact_sheet(
                        loaded.model,
                        case.trajectory,
                        lstm_datasets[loaded.run_id],
                        trajectory_indices[case.lesion_id],
                        output_path,
                        device,
                    )
                else:
                    embedding_id = (
                        int(patient_to_idx[case.trajectory.patient_id]) * 100
                        + int(case.trajectory.label_id)
                    )
                    _plot_continuous_contact_sheet(
                        loaded.model,
                        case.trajectory,
                        embedding_id,
                        output_path,
                        device,
                    )
                if normal_plot_count == 5:
                    break

        model_metrics = _summary_metrics(rows, loaded.family)
        if model_metrics:
            mlflow.log_metrics(model_metrics)

        if (
            args.skip_embedding_test
            or loaded.family not in EMBEDDING_SUPPORTED_FAMILIES
        ):
            continue

        if embedding_trajectory is None or embedding_model_idx is None:
            raise RuntimeError("Embedding trajectory setup was not completed.")
        embedding_case = EvaluationCase(
            f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]}",
            embedding_trajectory,
            "trajectory",
            embedding_trajectory_dates[-1],
            None,
            tuple(embedding_trajectory_dates),
        )
        embedding_path = _embedding_cache_path(
            loaded, args, embedding_trajectory_dates, device
        )
        if embedding_path.is_file():
            print(f"Loading cached {loaded.family} embedding from {embedding_path}.")
        else:
            print(
                f"Fitting frozen-network embedding for {loaded.family} on "
                f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]} using "
                f"{len(embedding_trajectory_dates)} timepoint(s) and "
                f"{args.embedding_samples_per_epoch} sampled example(s) per epoch."
            )

        def fit_embedding() -> torch.Tensor:
            _seed_embedding_fit(args.embedding_seed)
            return _fit_embedding(
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

        fitted_embedding = _load_or_fit_embedding(
            embedding_path, device, fit_embedding
        )
        fitted_embedding_key = _embedding_fingerprint(fitted_embedding)
        if args.plot_contact_sheets:
            output_path = args.output.with_name(
                f"{args.output.stem}_contact_sheets"
            ) / f"{loaded.family}_embedding_{embedding_case.lesion_id}.png"
            if loaded.family == "lstm":
                _plot_lstm_contact_sheet(
                    loaded.model,
                    embedding_trajectory,
                    embedding_lstm_datasets[loaded.run_id],
                    0,
                    output_path,
                    device,
                    embedding_override=fitted_embedding,
                )
            else:
                _plot_continuous_contact_sheet(
                    loaded.model,
                    embedding_trajectory,
                    embedding_model_idx,
                    output_path,
                    device,
                    embedding_override=fitted_embedding,
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
            if loaded.family == "lstm":
                infer = lambda target_volume: _infer_lstm(
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
                infer = lambda target_volume: _infer_continuous_model(
                    loaded,
                    target_volume,
                    embedding_model_idx,
                    args.resolution,
                    args.chunk_size,
                    device,
                    embedding_override=fitted_embedding,
                )
            adapted_sums = cached_result(
                f"{loaded.family}_embedding",
                loaded.run_id,
                trajectory_case,
                infer,
                inference_key=fitted_embedding_key,
            )
            record_result(
                f"{loaded.family}_embedding",
                loaded.run_id,
                trajectory_case,
                adapted_sums,
            )

        embedding_metrics = _summary_metrics(rows, f"{loaded.family}_embedding")
        if embedding_metrics:
            mlflow.log_metrics(embedding_metrics)

    for model_name in DILATION_BASELINE_NAMES:
        for case in _print_progress(
            cases, f"Evaluating {model_name}", "targets"
        ):
            growth_rate = (
                global_growth_rate
                if model_name == "dilation_global"
                else lesion_growth_rates[case.lesion_id]
            )
            sums = cached_result(
                model_name,
                "computed",
                case,
                lambda target_volume: _infer_dilation_baseline(
                    case,
                    target_volume,
                    growth_rate,
                    args.resolution,
                    args.chunk_size,
                ),
                growth_rate=growth_rate,
            )
            record_result(
                model_name,
                "computed",
                case,
                sums,
                growth_rate=growth_rate,
            )
        baseline_metrics = _summary_metrics(rows, model_name)
        if baseline_metrics:
            mlflow.log_metrics(baseline_metrics)
        if (
            not args.skip_embedding_test
            and embedding_trajectory is not None
            and embedding_dilation_growth_rate is not None
        ):
            embedding_growth_rate = (
                global_growth_rate
                if model_name == "dilation_global"
                else embedding_dilation_growth_rate
            )
            embedding_model_name = f"{model_name}_embedding"
            for date_index, date in enumerate(
                _print_progress(
                    embedding_trajectory_dates,
                    f"Evaluating {embedding_model_name}",
                    "timepoints",
                )
            ):
                embedding_case = EvaluationCase(
                    f"{EMBEDDING_TRAJECTORY[0]}_{EMBEDDING_TRAJECTORY[1]}",
                    embedding_trajectory,
                    "trajectory",
                    date,
                    None,
                    tuple(embedding_trajectory_dates[:date_index]) or (date,),
                )
                sums = cached_result(
                    embedding_model_name,
                    "computed",
                    embedding_case,
                    lambda target_volume: _infer_dilation_baseline(
                        embedding_case,
                        target_volume,
                        embedding_growth_rate,
                        args.resolution,
                        args.chunk_size,
                    ),
                    growth_rate=embedding_growth_rate,
                )
                record_result(
                    embedding_model_name,
                    "computed",
                    embedding_case,
                    sums,
                    growth_rate=embedding_growth_rate,
                )
            embedding_metrics = _summary_metrics(rows, embedding_model_name)
            if embedding_metrics:
                mlflow.log_metrics(embedding_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-target results written to {args.output}")
    model_names_to_summarize = []
    for loaded in models:
        model_names = [loaded.family]
        if (
            not args.skip_embedding_test
            and loaded.family in EMBEDDING_SUPPORTED_FAMILIES
        ):
            model_names.append(f"{loaded.family}_embedding")
        model_names_to_summarize.extend(model_names)
    model_names_to_summarize.extend(DILATION_BASELINE_NAMES)
    if not args.skip_embedding_test and embedding_trajectory is not None:
        model_names_to_summarize.extend(
            f"{name}_embedding" for name in DILATION_BASELINE_NAMES
        )
    for model_name in model_names_to_summarize:
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
                if split == "extrapolation":
                    weights = _extrapolation_weights(split_rows)
                    print(
                        f"{model_name} extrapolation weights: "
                        + ", ".join(
                            f"{lesion_id}={weight:.4g}"
                            for lesion_id, weight in sorted(weights.items())
                        )
                    )
                dice_score = float(
                    _average_metric(split_rows, "dice_score")
                )
                volume_errors = [
                    row["wrongly_predicted_volume_cm3"] for row in split_rows
                ]
                validation_params.update(
                    {
                        f"{prefix}_dice_loss": str(1.0 - dice_score),
                        f"{prefix}_dice_score": str(dice_score),
                        f"{prefix}_median_wrongly_predicted_volume_cm3": str(
                            float(np.median(volume_errors))
                        ),
                        f"{prefix}_average_wrongly_predicted_volume_cm3": str(
                            _average_metric(
                                split_rows, "wrongly_predicted_volume_cm3"
                            )
                        ),
                        f"{prefix}_wrongly_predicted_volume_cm3": str(
                            float(np.sum(volume_errors))
                        ),
                    }
                )
                print(
                    f"{model_name} {split}: "
                    f"dice loss={1.0 - dice_score:.5f}, "
                    f"dice score={dice_score:.5f}, "
                    f"median volume error={np.median(volume_errors):.5f} cm^3, "
                    f"average volume error="
                    f"{_average_metric(split_rows, 'wrongly_predicted_volume_cm3'):.5f} cm^3, "
                    f"total volume error={np.sum(volume_errors):.5f} cm^3, "
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