"""Patient-level plots for the patient-wise neural ODE experiment."""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import mlflow
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom

os.chdir(Path(__file__).resolve().parents[2])

from utils.paths import DatasetPaths


class PatientPlottingSample:
    def __init__(self, patient_id, embedding_id, trajectories):
        self.patient_id = patient_id
        self.embedding_id = embedding_id
        self.trajectories = trajectories


class PatientPlottingDataset:
    """Yield one plotting sample per patient represented in plotting_samples."""

    def __init__(self, trajectories):
        with open(DatasetPaths().patient_to_idx(), "r") as file:
            patient_to_idx = json.load(file)
        with open(DatasetPaths().plotting_samples(), "r") as file:
            selected_lesions = {
                f"{sample['Patient']}_{sample['Lesion']}" for sample in json.load(file)
            }

        grouped = {}
        selected_patients = set()
        for trajectory in trajectories:
            patient_id = trajectory.patient_id
            grouped.setdefault(patient_id, {})
            grouped[patient_id].setdefault(trajectory.label_id, trajectory)
            if f"{patient_id}_{trajectory.label_id}" in selected_lesions:
                selected_patients.add(patient_id)

        self.samples = [
            PatientPlottingSample(
                patient_id,
                patient_to_idx[patient_id],
                list(grouped[patient_id].values()),
            )
            for patient_id in sorted(selected_patients)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _allowed_dates(trajectory):
    dates = (
        trajectory.allowed_dates
        if hasattr(trajectory, "allowed_dates")
        else trajectory.dates
    )
    return {str(date) for date in dates}


def _load_mask(trajectory, date):
    path = Path(trajectory.path).parent / date / "trajectory.nii.gz"
    image = nib.load(str(path))
    return np.isin(image.get_fdata(), [trajectory.label_id]), image.affine


def _load_patient_union(sample, date, expected_shape=None, expected_affine=None):
    labels = None
    affine = expected_affine

    for trajectory in sample.trajectories:
        if date not in _allowed_dates(trajectory):
            continue
        lesion_mask, lesion_affine = _load_mask(trajectory, date)
        if labels is None:
            labels = np.zeros(lesion_mask.shape, dtype=bool)
            if (
                expected_shape is not None
                and lesion_mask.shape != tuple(expected_shape)
            ):
                raise ValueError(f"Label shape mismatch for patient {sample.patient_id}.")
            if affine is None:
                affine = lesion_affine
            elif not np.allclose(lesion_affine, affine):
                raise ValueError(f"Label affine mismatch for patient {sample.patient_id}.")
        elif lesion_mask.shape != labels.shape or not np.allclose(
            lesion_affine, affine
        ):
            raise ValueError(f"Label geometry mismatch for patient {sample.patient_id}.")
        np.logical_or(labels, lesion_mask, out=labels)

    if labels is None or affine is None:
        raise ValueError(
            f"No trajectory masks found for patient {sample.patient_id} on {date}."
        )
    return labels, affine


def _patient_first_date(sample):
    return min(
        datetime.strptime(str(date), "%Y-%m-%d")
        for trajectory in sample.trajectories
        for date in trajectory.dates
    )


def _selected_dates(sample, max_dates=12):
    dates = sorted(
        set().union(*(_allowed_dates(trajectory) for trajectory in sample.trajectories))
    )
    if not dates:
        raise ValueError(f"Patient {sample.patient_id} has no allowed plotting dates.")
    if len(dates) > max_dates:
        indices = np.unique(np.linspace(0, len(dates) - 1, max_dates).round().astype(int))
        dates = [dates[index] for index in indices]
    return dates


def _slice_positions(sample, shape):
    positions = set()
    for trajectory in sample.trajectories:
        union = None
        for date in sorted(_allowed_dates(trajectory)):
            mask, _ = _load_mask(trajectory, date)
            if mask.shape != tuple(shape):
                raise ValueError(
                    f"Label shape mismatch for lesion "
                    f"{trajectory.patient_id}_{trajectory.label_id}."
                )
            if union is None:
                union = mask
            else:
                np.logical_or(union, mask, out=union)
        if union is not None and np.any(union):
            positions.add(int(np.median(np.argwhere(union)[:, 2])))

    if not positions:
        raise ValueError(f"Patient {sample.patient_id} has no labeled voxels to plot.")
    return sorted(positions)


def _predict_slice(model, patient_idx, affine, shape, z, day, side_length, device):
    grid_i, grid_j = np.meshgrid(
        np.linspace(0, shape[0] - 1, side_length),
        np.linspace(0, shape[1] - 1, side_length),
        indexing="ij",
    )
    voxel_coords = np.column_stack(
        [
            grid_i.reshape(-1),
            grid_j.reshape(-1),
            np.full(grid_i.size, z),
            np.ones(grid_i.size),
        ]
    )
    physical_coords = (affine @ voxel_coords.T).T[:, :3]
    center_voxel = (np.asarray(shape) - 1) / 2.0
    center_physical = affine[:3, :3] @ center_voxel + affine[:3, 3]
    spatial_coords = (physical_coords - center_physical) / 100.0
    coords = np.column_stack(
        [spatial_coords, np.full(len(spatial_coords), day, dtype=np.float32)]
    ).astype(np.float32)
    coords = torch.from_numpy(coords).unsqueeze(0).to(device)
    patient_tensor = torch.tensor([patient_idx], dtype=torch.long, device=device)

    probabilities = []
    with torch.no_grad():
        for start in range(0, coords.shape[1], 8192):
            logits = model(coords[:, start : start + 8192], patient_tensor)
            probabilities.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
    return np.concatenate(probabilities).reshape(side_length, side_length)


def _build_plot_data(model, epoch, sample, config):
    dates = _selected_dates(sample)
    first_date = _patient_first_date(sample)
    labels_by_date = []
    affine = None
    shape = None

    for date in dates:
        labels, date_affine = _load_patient_union(
            sample, date, expected_shape=shape, expected_affine=affine
        )
        if shape is None:
            shape = labels.shape
        if affine is None:
            affine = date_affine
        elif not np.allclose(date_affine, affine):
            raise ValueError(f"Label affine mismatch for patient {sample.patient_id}.")
        labels_by_date.append(labels)

    z_positions = _slice_positions(sample, shape)
    side_length = int(getattr(config, "patient_plot_side_length", 96))
    if side_length < 2:
        raise ValueError("patient_plot_side_length must be at least 2.")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    predictions_by_date = []
    try:
        for date in dates:
            day = (
                datetime.strptime(date, "%Y-%m-%d") - first_date
            ).days
            predictions_by_date.append(
                [
                    _predict_slice(
                        model,
                        sample.embedding_id,
                        affine,
                        shape,
                        z,
                        day,
                        side_length,
                        device,
                    )
                    for z in z_positions
                ]
            )
    finally:
        if was_training:
            model.train()

    return {
        "epoch": epoch,
        "sample": sample,
        "dates": dates,
        "days": [
            (datetime.strptime(date, "%Y-%m-%d") - first_date).days
            for date in dates
        ],
        "z_positions": z_positions,
        "labels": labels_by_date,
        "predictions": predictions_by_date,
    }


_PLOT_CACHE = {}


def _cache_key(model, epoch, sample):
    return id(model), epoch, sample.patient_id


def _get_plot_data(model, epoch, sample, config):
    key = _cache_key(model, epoch, sample)
    if key not in _PLOT_CACHE:
        _PLOT_CACHE[key] = _build_plot_data(model, epoch, sample, config)
    return key, _PLOT_CACHE[key]


def _draw_slice(ax, prediction, labels, z, day):
    ax.imshow(prediction, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    if labels.any():
        resized_labels = zoom(
            labels.astype(float),
            (
                prediction.shape[0] / labels.shape[0],
                prediction.shape[1] / labels.shape[1],
            ),
            order=0,
        )
        ax.contour(
            resized_labels, levels=[0.5], colors=["red"], linewidths=1
        )
    if prediction.min() < 0.5 < prediction.max():
        ax.contour(prediction, levels=[0.5], colors=["lime"], linewidths=1)
    ax.set_title(f"z={z}, day={day}")
    ax.axis("off")


def _artifact_path(sample, epoch, suffix):
    patient_id = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in sample.patient_id
    )
    return Path(tempfile.gettempdir()) / (
        f"{epoch:04d}_patient_union_{patient_id}_{sample.embedding_id}.{suffix}"
    )


def _log_artifact(path):
    try:
        mlflow.log_artifact(str(path))
    finally:
        path.unlink(missing_ok=True)


def plot_patient_contact_sheet(model, epoch, sample, config, final_side_length=False):
    del final_side_length
    key, data = _get_plot_data(model, epoch, sample, config)
    dates = data["dates"]
    z_positions = data["z_positions"]
    fig, axes = plt.subplots(
        len(dates),
        len(z_positions),
        figsize=(3 * len(z_positions), 2.6 * len(dates)),
        squeeze=False,
    )

    for row, day in enumerate(data["days"]):
        for column, z in enumerate(z_positions):
            _draw_slice(
                axes[row, column],
                data["predictions"][row][column],
                np.take(data["labels"][row], z, axis=2),
                z,
                day,
            )
    fig.suptitle(
        f"Patient {sample.patient_id}: union label (red), prediction (green)",
        y=1.0,
    )
    fig.tight_layout()
    path = _artifact_path(sample, epoch, "png")
    try:
        fig.savefig(path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)
    _log_artifact(path)
    _PLOT_CACHE[key] = data


def plot_patient_time_evolution(model, epoch, sample, config, final_side_length=False):
    del final_side_length
    key, data = _get_plot_data(model, epoch, sample, config)
    fig, axes = plt.subplots(
        1,
        len(data["z_positions"]),
        figsize=(3 * len(data["z_positions"]), 3),
        squeeze=False,
    )

    def update(frame):
        for column, z in enumerate(data["z_positions"]):
            axes[0, column].clear()
            _draw_slice(
                axes[0, column],
                data["predictions"][frame][column],
                np.take(data["labels"][frame], z, axis=2),
                z,
                data["days"][frame],
            )
        fig.suptitle(
            f"Patient {sample.patient_id} - day {data['days'][frame]} "
            "(union label in red, prediction in green)"
        )
        return []

    animation = FuncAnimation(fig, update, frames=len(data["dates"]), blit=False)
    path = _artifact_path(sample, epoch, "gif")
    try:
        animation.save(path, writer=PillowWriter(fps=2))
    finally:
        plt.close(fig)
        _PLOT_CACHE.pop(key, None)
    _log_artifact(path)
