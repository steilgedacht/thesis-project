"""Create a paper-style lesion growth timeline and MRI contact sheet."""

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter, LinearLocator
import mlflow
from mlflow.pytorch import load_model
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import torch


ROBOTO_FONT = Path(__file__).resolve().parent / "fonts" / "Roboto[wdth,wght].ttf"
font_manager.fontManager.addfont(str(ROBOTO_FONT))
plt.rcParams["font.family"] = "Roboto"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.dataloader import Plotting_LesionDataset
from utils.lesion_trajectory import Lesion_Trajectory
from utils.mri_dataloader import MRI_Dataloader
from utils.model_inr import LesionINR
from utils.model_lstm import LesionLSTM, LesionLatentLSTM
from utils.model_neural_ode import NeuralODE_INR
from utils.model_neural_ode_translation import NeuralODETranslationINR
from utils.patient import Patient
from utils.paths import DatasetPaths
from utils.train_plotting import (
    _compute_lesion_z_voxel,
    _compute_reference_geometry,
    _resize_2d,
    _voxel_to_model_space,
)


RUN_ID = "50d82532807e4da3ab4eead3869afb8c"
TRACKING_URI = "http://127.0.0.1:5124"
MODEL_ARTIFACT = "lesion_ode_model"
LABEL_COLOR = "#c62523"
PREDICTION_COLOR = "#34c759"
INTERPOLATION_COLOR = "#B8923A"
EXTRAPOLATION_COLOR = INTERPOLATION_COLOR
EVALUATION_LABEL_COLOR = "#CB9412"
TIMELINE_COLOR = PREDICTION_COLOR
INR_COLOR = "#8e44ad"
LSTM_COLOR = "#089284"
DEFAULT_INR_RUN_ID = "ac4338b4ad0a4f5b8b72d48b159520ca"
DEFAULT_LSTM_RUN_ID = "8d67b8dd7050458fa6ea391e46296639"
PLOT_CACHE_VERSION = 3


class _OrthogonalConnection:
    def __init__(self, lane_index=0, lane_count=1):
        self.lane_fraction = 0.4 - 0.2 * lane_index / max(lane_count - 1, 1) 

    def __call__(self, pos_a, pos_b, **kwargs):
        elbow_y = pos_a[1] + self.lane_fraction * (pos_b[1] - pos_a[1])
        return MplPath(
            [
                pos_a,
                (pos_a[0], elbow_y),
                (pos_b[0], elbow_y),
                pos_b,
            ],
            [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
        )


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--model-artifact", default=MODEL_ARTIFACT)
    parser.add_argument("--inr-run-id", default=DEFAULT_INR_RUN_ID)
    parser.add_argument("--inr-model-artifact", default="lesion_inr_model")
    parser.add_argument("--lstm-run-id", default=DEFAULT_LSTM_RUN_ID)
    parser.add_argument("--lstm-model-artifact", default="lesion_lstm_autoencoder")
    parser.add_argument(
        "--sample",
        help="Plot only one sample, e.g. YG_Q1FS2D6EVG3E_lesion_9.png.",
    )
    parser.add_argument(
        "--node-only",
        "--no-comparisons",
        action="store_true",
        dest="no_comparisons",
        help="Plot only the Neural ODE (NODE), skipping INR and LSTM comparisons.",
    )
    parser.add_argument("--time-steps", type=int, default=300,
                        help="Number of ODE evaluation times before adding label dates.")
    parser.add_argument("--prediction-spacing-mm", type=float, default=6.0,
                        help="Physical spacing of the 3D quadrature grid for volume estimates.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments/11_paper/plot_main_page_samples",
    )
    args = parser.parse_args()
    if args.time_steps < 2:
        parser.error("--time-steps must be at least 2")
    if args.prediction_spacing_mm <= 0:
        parser.error("--prediction-spacing-mm must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def _plotting_dataset():
    dataloader = MRI_Dataloader()
    dataloader.cache_lesion_trajectories_from_n_scans(n_scans=6, only_growing=True)
    trajectories = dataloader.cache_lesion_trajectories
    if trajectories is None:
        raise RuntimeError("MRI_Dataloader did not populate its lesion trajectory cache.")
    with open(DatasetPaths().validation_samples(), encoding="utf-8") as file:
        validation_samples = json.load(file)
    sample_order = {
        f"{sample['Patient']}_{sample['Lesion']}": index
        for index, sample in enumerate(validation_samples)
    }
    trajectories = [
        trajectory
        for trajectory in trajectories
        if f"{trajectory.patient_id}_{trajectory.label_id}" in sample_order
    ]
    trajectories.sort(
        key=lambda item: sample_order.get(
            f"{item.patient_id}_{item.label_id}",
            len(sample_order),
        )
    )
    plotting_dataset = Plotting_LesionDataset(trajectories, device="cpu")
    plotting_dataset.trajectories = trajectories
    return dataloader, plotting_dataset


def _interpolation_date(trajectory: Lesion_Trajectory) -> str | None:
    with open(DatasetPaths().validation_samples(), encoding="utf-8") as file:
        validation_samples = json.load(file)
    sample = next(
        (
            item for item in validation_samples
            if item["Patient"] == trajectory.patient_id
            and str(item["Lesion"]) == str(trajectory.label_id)
        ),
        None,
    )
    if sample is None:
        return None
    dates = trajectory.allowed_dates
    index = int(sample["interpolation"])
    if not 0 <= index < len(dates):
        raise ValueError(
            f"Interpolation index {index} is outside the {len(dates)} dates "
            f"for {trajectory.patient_id}, lesion {trajectory.label_id}."
        )
    return dates[index]


def _trajectory_data(trajectory: Lesion_Trajectory):
    labels_with_days, affine = trajectory.load_labels_for_inr(
        absolute_day_number=True,
        affine=True,
    )
    if not labels_with_days:
        raise ValueError("The selected lesion trajectory has no label volumes.")

    labels = [volume for volume, _ in labels_with_days]
    label_days = np.asarray([day for _, day in labels_with_days], dtype=float)
    shape = np.asarray(labels[0].shape, dtype=int)
    if any(volume.shape != tuple(shape) for volume in labels):
        raise ValueError("Label volumes in the selected trajectory have different shapes.")

    geometry = _compute_reference_geometry(shape, affine)
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    if voxel_volume_mm3 <= 0:
        raise ValueError("The selected trajectory has an invalid voxel affine.")

    lesion_z = int(np.clip(round(_compute_lesion_z_voxel(labels_with_days)), 0, shape[2] - 1))
    label_volumes_mm3 = np.asarray(
        [np.count_nonzero(volume) * voxel_volume_mm3 for volume in labels],
        dtype=float,
    )
    label_radii_mm = _radius_from_volume_mm3(label_volumes_mm3)
    return labels, label_days, affine, geometry, shape, lesion_z, label_radii_mm


def _radius_from_volume_mm3(volume_mm3: np.ndarray) -> np.ndarray:
    """Radius of a sphere with the supplied volume."""
    return np.cbrt(3.0 * np.maximum(volume_mm3, 0.0) / (4.0 * np.pi))


def _representative_label_indices(size_mm3, interpolation_index=None, max_labels=6,
                                  minimum_change_mm3=100.0):
    """Keep endpoints, the evaluation label, and labels after larger size changes."""
    if len(size_mm3) <= max_labels:
        return np.arange(len(size_mm3))

    selected = {0, len(size_mm3) - 1}
    if interpolation_index is not None:
        selected.add(interpolation_index)
    changes = np.abs(np.diff(size_mm3))
    candidates = sorted(
        range(1, len(size_mm3)),
        key=lambda index: changes[index - 1],
        reverse=True,
    )
    for index in candidates:
        if len(selected) >= max_labels:
            break
        if changes[index - 1] >= minimum_change_mm3:
            selected.add(index)
    for index in candidates:
        if len(selected) >= max_labels:
            break
        selected.add(index)
    return np.asarray(sorted(selected), dtype=int)


def _volume_from_radius_mm(radius_mm):
    return (4.0 / 3.0) * np.pi * np.asarray(radius_mm, dtype=float) ** 3


def _spatial_grid(shape, affine, geometry, target_spacing_mm):
    axis_spacing_mm = np.linalg.norm(affine[:3, :3], axis=0)
    n_per_axis = [
        max(2, int(np.ceil((size - 1) * spacing / target_spacing_mm)) + 1)
        if size > 1 else 1
        for size, spacing in zip(shape, axis_spacing_mm)
    ]
    axes = [
        np.linspace(0, size - 1, count, dtype=np.float32)
        for size, count in zip(shape, n_per_axis)
    ]
    voxel_grid = np.stack(
        [axis.ravel() for axis in np.meshgrid(*axes, indexing="ij")],
        axis=1,
    )
    model_xyz = _voxel_to_model_space(
        voxel_grid,
        affine,
        geometry.volume_center_phys,
    )
    total_volume_mm3 = float(np.prod(shape) * abs(np.linalg.det(affine[:3, :3])))
    sample_weight_mm3 = total_volume_mm3 / len(model_xyz)
    return model_xyz, sample_weight_mm3


def _encode_chunks(model, model_xyz, device, batch_size, translation=None):
    model_dtype = next(model.parameters()).dtype
    return [
        model.spatial_encoding(
            torch.as_tensor(
                model_xyz[start:start + batch_size],
                dtype=model_dtype,
                device=device,
            ).unsqueeze(0)
            - (translation.unsqueeze(1) if translation is not None else 0)
        )
        for start in range(0, len(model_xyz), batch_size)
    ]


def _ode_states(model, patient_idx, time_points):
    patient_embedding = model.patient_embedding(patient_idx)
    initial_state = model.initial_state_net(patient_embedding)
    initial_translation = None
    if isinstance(model, NeuralODETranslationINR):
        initial_translation = model.initial_translation_net(patient_embedding)
    states = []
    for day in time_points:
        normalized_time = torch.as_tensor(
            [[float(day) / model.max_t]],
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        shape_state = (
            model.shape_ode_solver(initial_state, normalized_time, patient_embedding)
            if isinstance(model, NeuralODETranslationINR)
            else model.ode_solver(initial_state, normalized_time, patient_embedding)
        )
        if isinstance(model, NeuralODETranslationINR):
            if initial_translation is None:
                raise RuntimeError("Translation state was not initialized.")
            translation = model.translation_ode_solver(
                initial_translation,
                normalized_time,
                patient_embedding,
            )
            states.append((shape_state, translation))
        else:
            states.append(shape_state)
    return states


def _predicted_volumes_mm3(
    model,
    encoded_chunks,
    states,
    sample_weight_mm3,
    model_xyz=None,
    batch_size=8192,
):
    volumes = []
    with torch.inference_mode():
        for state in states:
            if isinstance(state, tuple):
                shape_state, translation = state
                if model_xyz is None:
                    raise ValueError("Coordinates are required for translated model states.")
                state_encoded_chunks = _encode_chunks(
                    model,
                    model_xyz,
                    translation.device,
                    batch_size,
                    translation,
                )
            else:
                shape_state = state
                state_encoded_chunks = encoded_chunks
            predicted_volume = 0.0
            for encoded in state_encoded_chunks:
                state_for_chunk = shape_state.unsqueeze(1).expand(-1, encoded.shape[1], -1)
                logits = model.decoder(encoded, state_for_chunk)
                predicted_volume += float(torch.sigmoid(logits).sum().item())
            volumes.append(predicted_volume * sample_weight_mm3)
    return np.asarray(volumes, dtype=float)


def _voxel_model_grid(shape, affine, geometry, z_index, max_side_length=192):
    scale = min(1.0, max_side_length / max(shape[:2]))
    grid_shape = tuple(max(1, int(round(size * scale))) for size in shape[:2])
    axis_i = np.linspace(0, shape[0] - 1, grid_shape[0], dtype=np.float32)
    axis_j = np.linspace(0, shape[1] - 1, grid_shape[1], dtype=np.float32)
    grid_i, grid_j = np.meshgrid(
        axis_i,
        axis_j,
        indexing="ij",
    )
    voxels = np.stack(
        [grid_i.ravel(), grid_j.ravel(), np.full(grid_i.size, z_index, dtype=np.float32)],
        axis=1,
    )
    return _voxel_to_model_space(voxels, affine, geometry.volume_center_phys), grid_shape


def _prediction_slice_masks(model, encoded_chunks, state):
    if isinstance(state, tuple):
        state = state[0]
    masks = []
    with torch.inference_mode():
        for encoded in encoded_chunks:
            state_for_chunk = state.unsqueeze(1).expand(-1, encoded.shape[1], -1)
            logits = model.decoder(encoded, state_for_chunk)
            masks.append((logits.squeeze(0).squeeze(-1) >= 0).cpu().numpy())
    return np.concatenate(masks).reshape(-1)


def _inr_prediction_volumes_mm3(model, patient_idx, model_xyz, time_points,
                                sample_weight_mm3, batch_size):
    model_dtype = next(model.parameters()).dtype
    volumes = []
    with torch.inference_mode():
        for day in time_points:
            time_column = np.full((len(model_xyz), 1), float(day), dtype=np.float32)
            coords = np.concatenate([model_xyz, time_column], axis=1)
            predicted_voxels = 0
            for start in range(0, len(coords), batch_size):
                coords_tensor = torch.as_tensor(
                    coords[start:start + batch_size],
                    dtype=model_dtype,
                    device=patient_idx.device,
                ).unsqueeze(0)
                logits = model(coords_tensor, patient_idx)
                predicted_voxels += int((logits >= 0).sum().item())
            volumes.append(predicted_voxels * sample_weight_mm3)
    return np.asarray(volumes, dtype=float)


def _lstm_grid_size(model):
    if isinstance(model, LesionLSTM):
        return tuple(model.grid_size)
    return tuple(model.decoder.out_size)


def _lstm_plotting_trajectory(trajectory, model):
    from utils.dataloader_lstm import LesionSequenceDataset

    sequence_dataset = LesionSequenceDataset(
        [trajectory],
        grid_size=_lstm_grid_size(model),
        margin_mm=20.0,
        min_extent_mm=80.0,
        device="cpu",
    )
    dates = list(
        trajectory.allowed_dates if hasattr(trajectory, "allowed_dates") else trajectory.dates
    )
    if not dates:
        raise ValueError(
            f"No dates available for LSTM sequence data for {trajectory.patient_id}, "
            f"lesion {trajectory.label_id}."
        )
    cached = sequence_dataset._load_and_resample_trajectory(0, dates)
    trajectory.embedding_id = sequence_dataset.get_embedding_id(trajectory)
    trajectory.grids = np.stack([cached["grids"][date] for date in dates], axis=0)[:, None]
    trajectory.times = np.asarray([cached["times"][date] for date in dates], dtype=np.float32)
    trajectory.canonical_center = cached["center"]
    trajectory.canonical_extent_mm = cached["extent"]
    return trajectory


def _latent_lstm_prediction(model, observed_latents, observed_times,
                            target_time, patient_embedding):
    hidden = None
    if len(observed_times) > 1:
        history_inputs = []
        for index in range(len(observed_times) - 1):
            time = torch.as_tensor(
                [observed_times[index + 1]],
                dtype=observed_latents.dtype,
                device=observed_latents.device,
            )
            time_encoding = model.time_encoder(time)
            features = [observed_latents[:, index], time_encoding]
            if patient_embedding is not None:
                features.append(patient_embedding)
            history_inputs.append(torch.cat(features, dim=-1))
        _, hidden = model.rnn(torch.stack(history_inputs, dim=1))

    time = torch.as_tensor(
        [target_time],
        dtype=observed_latents.dtype,
        device=observed_latents.device,
    )
    features = [observed_latents[:, -1], model.time_encoder(time)]
    if patient_embedding is not None:
        features.append(patient_embedding)
    output, _ = model.rnn(torch.cat(features, dim=-1).unsqueeze(1), hidden)
    predicted_latent = model.fc_out(output[:, -1])
    return torch.sigmoid(model.decoder(predicted_latent))


def _lstm_prediction_volumes_mm3(model, trajectory, time_points):
    trajectory = _lstm_plotting_trajectory(trajectory, model)
    device = next(model.parameters()).device
    real_times = np.asarray(trajectory.times, dtype=np.float32)
    real_grids = torch.as_tensor(
        trajectory.grids,
        dtype=next(model.parameters()).dtype,
        device=device,
    ).unsqueeze(0)
    patient_idx = torch.tensor(
        [trajectory.embedding_id],
        dtype=torch.long,
        device=device,
    )
    if isinstance(model, LesionLatentLSTM):
        with torch.inference_mode():
            observed_latents = torch.stack(
                [
                    model.encoder(real_grids[:, index])
                    for index in range(real_grids.shape[1])
                ],
                dim=1,
            )
            patient_embedding = (
                model.patient_emb(patient_idx)
                if model.use_patient_embedding
                else None
            )
    else:
        observed_latents = None
        patient_embedding = None

    grid_size = _lstm_grid_size(model)
    sample_weight_mm3 = float(trajectory.canonical_extent_mm) ** 3 / float(
        np.prod(np.asarray(grid_size, dtype=float) - 1)
    )
    volumes = []
    with torch.inference_mode():
        for day in time_points:
            history_length = int(np.searchsorted(real_times, day, side="left"))
            if history_length == 0:
                predicted_grid = real_grids[0, 0]
            elif isinstance(model, LesionLatentLSTM):
                if observed_latents is None:
                    raise RuntimeError("LSTM observation encodings were not prepared.")
                predicted_grid = _latent_lstm_prediction(
                    model,
                    observed_latents[:, :history_length],
                    real_times[:history_length],
                    float(day),
                    patient_embedding,
                )[0, 0]
            else:
                history_grids = real_grids[:, :history_length]
                history_times = torch.as_tensor(
                    real_times[:history_length],
                    dtype=real_grids.dtype,
                    device=device,
                ).unsqueeze(0)
                target_time = torch.as_tensor(
                    [[float(day)]],
                    dtype=real_grids.dtype,
                    device=device,
                )
                predictions = model(
                    history_grids,
                    torch.cat([history_times, target_time], dim=1),
                    patient_idx,
                    teacher_forcing=True,
                    n_future=1,
                )
                predicted_grid = torch.sigmoid(predictions[0, -1, 0])
            predicted_voxels = int((predicted_grid > 0.5).sum().item())
            volumes.append(predicted_voxels * sample_weight_mm3)
    return np.asarray(volumes, dtype=float)


def _plot_cache_signature(args, trajectory):
    models = [
        {
            "name": "ODE",
            "run_id": args.run_id,
            "artifact": args.model_artifact,
        }
    ]
    if not args.no_comparisons:
        models.extend(
            [
                {
                    "name": "INR",
                    "run_id": args.inr_run_id,
                    "artifact": args.inr_model_artifact,
                },
                {
                    "name": "LSTM",
                    "run_id": args.lstm_run_id,
                    "artifact": args.lstm_model_artifact,
                },
            ]
        )
    return {
        "version": PLOT_CACHE_VERSION,
        "tracking_uri": args.tracking_uri,
        "patient_id": str(trajectory.patient_id),
        "lesion_id": str(trajectory.label_id),
        "dates": list(trajectory.allowed_dates),
        "models": models,
        "time_steps": args.time_steps,
        "prediction_spacing_mm": args.prediction_spacing_mm,
    }


def _plot_cache_path(output_path):
    return output_path.with_suffix(output_path.suffix + ".cache.json")


def _plot_data_path(output_path):
    return output_path.with_suffix(output_path.suffix + ".data.npz")


def _read_plot_cache(output_path, signature):
    cache_path = _plot_cache_path(output_path)
    data_path = _plot_data_path(output_path)
    if not cache_path.is_file() or not data_path.is_file():
        return None
    try:
        metadata = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("signature") != signature:
            return None
        with np.load(data_path, allow_pickle=False) as archive:
            plot_data = {key: archive[key] for key in archive.files}
        plot_data["dates"] = metadata["dates"]
        plot_data["interpolation_date"] = metadata["interpolation_date"]
        if len(plot_data["panel_indices"]) != min(6, len(metadata["dates"])):
            return None
        comparison_predictions = {}
        comparison_colors = {"INR": INR_COLOR, "LSTM": LSTM_COLOR}
        for index, comparison in enumerate(metadata["comparisons"]):
            name = comparison["name"]
            comparison_predictions[name] = (
                plot_data.pop(f"comparison_{index}_days"),
                plot_data.pop(f"comparison_{index}_radii_mm"),
                comparison_colors[name],
            )
        plot_data["comparison_predictions"] = comparison_predictions
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        print(f"Ignoring invalid plot cache data at {cache_path}: {error}")
        return None
    return plot_data


def _write_plot_cache(output_path, signature, plot_data):
    cache_path = _plot_cache_path(output_path)
    data_path = _plot_data_path(output_path)
    comparisons = []
    arrays = {}
    for key in (
        "label_slices",
        "label_days",
        "label_radii_mm",
        "prediction_days",
        "prediction_radii_mm",
        "mri_slices",
        "prediction_slices",
        "panel_indices",
    ):
        arrays[key] = np.asarray(plot_data[key])
    for index, (name, (days, radii, color)) in enumerate(
        plot_data["comparison_predictions"].items()
    ):
        comparisons.append({"name": name, "color": color})
        arrays[f"comparison_{index}_days"] = np.asarray(days)
        arrays[f"comparison_{index}_radii_mm"] = np.asarray(radii)
    metadata = {
        "signature": signature,
        "dates": plot_data["dates"],
        "interpolation_date": plot_data["interpolation_date"],
        "comparisons": comparisons,
    }

    temporary_data_path = data_path.with_name(data_path.name + ".tmp")
    with temporary_data_path.open("wb") as file:
        np.savez_compressed(file, **arrays)
    os.replace(temporary_data_path, data_path)

    temporary_cache_path = cache_path.with_name(cache_path.name + ".tmp")
    temporary_cache_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_cache_path, cache_path)


def _render_plot_data(trajectory, plot_data, args, output_path):
    figure = _make_figure(
        trajectory,
        plot_data["label_slices"],
        plot_data["label_days"],
        plot_data["label_radii_mm"],
        plot_data["prediction_days"],
        plot_data["prediction_radii_mm"],
        plot_data["mri_slices"],
        plot_data["prediction_slices"],
        plot_data["dates"],
        plot_data["panel_indices"],
        plot_data["interpolation_date"],
        output_path,
        args.prediction_spacing_mm,
        plot_data["comparison_predictions"],
    )
    plt.close(figure)
    print(f"Saved plot to {output_path}")


def _aligned_mri_slices(dataloader, trajectory, affine, shape, z_index):
    patient = Patient(trajectory.patient_id, dataloader=dataloader)
    samples_by_date = {sample.date: sample for sample in patient.samples}
    slices = []
    target = (tuple(shape), affine)
    for date in trajectory.allowed_dates:
        sample = samples_by_date.get(date)
        if sample is None:
            raise FileNotFoundError(
                f"No MRI scan found for {trajectory.patient_id} on {date}."
            )
        mri, mri_affine = sample.load_mri(zoomed=True, affine=True)
        if mri.shape != tuple(shape) or not np.allclose(mri_affine, affine):
            mri = np.asarray(
                resample_from_to(nib.Nifti1Image(mri, mri_affine), target, order=1).dataobj
            )
        if mri.shape != tuple(shape):
            raise ValueError(f"Could not align the MRI volume for {date} to its label grid.")
        slices.append(mri[:, :, z_index])
    return slices


def _add_contact_panels(fig, grid, label_slices, label_days, label_radii_mm,
                        mri_slices, prediction_slices, dates, interpolation_date,
                        columns=4):
    rows = int(np.ceil(len(label_slices) / columns))
    for index, (label_slice, day, radius, mri_slice, prediction, date) in enumerate(
        zip(label_slices, label_days, label_radii_mm, mri_slices, prediction_slices, dates)
    ):
        ax = fig.add_subplot(grid[index // columns, index % columns])
        low, high = np.percentile(mri_slice, [1, 99])
        if low == high:
            high = low + 1
        ax.imshow(mri_slice, cmap="gray", vmin=low, vmax=high, origin="upper")
        if np.any(label_slice):
            ax.contour(label_slice, levels=[0.5], colors=[LABEL_COLOR], linewidths=1.1)
        if np.any(prediction):
            ax.contour(prediction, levels=[0.5], colors=[PREDICTION_COLOR], linewidths=1.1)
        is_interpolation = date == interpolation_date
        ax.axis("off")
        if is_interpolation:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(INTERPOLATION_COLOR)
                spine.set_linewidth(2.5)
        ax.text(
            0.03,
            0.04,
            f"t={day:.0f} days    r={radius:.1f} mm",
            transform=ax.transAxes,
            color="grey",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 1, "edgecolor": "none", "pad": 2},
        )
    for index in range(len(label_slices), rows * columns):
        fig.add_subplot(grid[index // columns, index % columns]).axis("off")


def _add_boundary_zoom_panels(fig, grid, label_slices, label_days,
                              label_radii_mm, mri_slices, prediction_slices,
                              columns=4):
    rows = int(np.ceil(len(label_slices) / columns))
    for index, (label_slice, day, radius, mri_slice, prediction) in enumerate(
        zip(label_slices, label_days, label_radii_mm, mri_slices, prediction_slices)
    ):
        ax = fig.add_subplot(grid[index // columns, index % columns])
        low, high = np.percentile(mri_slice, [1, 99])
        if low == high:
            high = low + 1
        ax.imshow(mri_slice, cmap="gray", vmin=low, vmax=high, origin="upper")
        if np.any(label_slice):
            ax.contour(label_slice, levels=[0.5], colors=[LABEL_COLOR], linewidths=1.1)
        if np.any(prediction):
            ax.contour(prediction, levels=[0.5], colors=[PREDICTION_COLOR], linewidths=1.1)

        boundary = np.asarray(label_slice, dtype=bool) | np.asarray(prediction, dtype=bool)
        boundary_rows, boundary_columns = np.nonzero(boundary)
        height, width = boundary.shape
        if len(boundary_rows):
            row_min, row_max = int(boundary_rows.min()), int(boundary_rows.max())
            column_min, column_max = int(boundary_columns.min()), int(boundary_columns.max())
            center_row = (row_min + row_max) / 2
            center_column = (column_min + column_max) / 2
            lesion_size = max(row_max - row_min + 1, column_max - column_min + 1)
            half_size = min(max(16, int(np.ceil(lesion_size * 1.5))), height / 2, width / 2)
        else:
            center_row, center_column = height / 2, width / 2
            half_size = min(height, width) / 4
        center_row = np.clip(center_row, half_size, height - half_size)
        center_column = np.clip(center_column, half_size, width - half_size)
        ax.set_xlim(center_column - half_size, center_column + half_size)
        ax.set_ylim(center_row + half_size, center_row - half_size)
        ax.axis("off")
        ax.text(
            0.03,
            0.04,
            f"t={day:.0f} days    r={radius:.1f} mm",
            transform=ax.transAxes,
            color="grey",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 1, "edgecolor": "none", "pad": 2},
        )
    for index in range(len(label_slices), rows * columns):
        fig.add_subplot(grid[index // columns, index % columns]).axis("off")


def _make_figure(trajectory, label_slices, label_days, label_radii_mm,
                 prediction_days, prediction_radii_mm, mri_slices,
                 prediction_slices, dates, panel_indices, interpolation_date,
                 output_path, prediction_spacing_mm, comparison_predictions=None):
    columns = 6
    if len(label_slices) > columns:
        raise ValueError(f"Expected at most {columns} MRI overlay panels, got {len(label_slices)}.")
    contact_rows = 1
    fig = plt.figure(figsize=(16, 7.92))
    outer = fig.add_gridspec(
        2,
        columns,
        height_ratios=[1.36, 2.6],
        hspace=0.16,
        wspace=0.035,
        top=0.94,
    )
    contact_rows_grid = outer[1, :].subgridspec(
        2,
        1,
        height_ratios=[1, 1],
        hspace=0,
        wspace=0.035,
    )
    contact_grid = contact_rows_grid[0, 0].subgridspec(
        contact_rows,
        columns,
        wspace=0.035,
    )
    _add_contact_panels(
        fig,
        contact_grid,
        label_slices,
        label_days[panel_indices],
        label_radii_mm[panel_indices],
        mri_slices,
        prediction_slices,
        [dates[index] for index in panel_indices],
        interpolation_date,
        columns,
    )
    zoom_grid = contact_rows_grid[1, 0].subgridspec(
        contact_rows,
        columns,
        wspace=0.035,
    )
    _add_boundary_zoom_panels(
        fig,
        zoom_grid,
        label_slices,
        label_days[panel_indices],
        label_radii_mm[panel_indices],
        mri_slices,
        prediction_slices,
        columns,
    )
    ax = fig.add_subplot(outer[0, :])

    end_day = float(label_days[-1])
    extrapolation_start = max(float(prediction_days[0]), end_day - 180.0)
    prediction_radii_mm = np.asarray(prediction_radii_mm, dtype=float)
    label_radii_mm = np.asarray(label_radii_mm, dtype=float)
    if end_day > extrapolation_start:
        ax.axvspan(
            extrapolation_start,
            end_day,
            color=EVALUATION_LABEL_COLOR,
            alpha=0.1,
            zorder=0,
            ymax=0.86
        )
    def plot_prediction_curve(days, radii, color, label, linewidth):
        for in_phase, linestyle in (
            (days < extrapolation_start, "-"),
            (days >= extrapolation_start, "-."),
        ):
            phase_days = days[in_phase]
            phase_radii = radii[in_phase]
            if len(phase_days) == 0:
                continue
            ax.fill_between(
                phase_days,
                0,
                phase_radii,
                color=color,
                alpha=0.08,
                zorder=1,
            )
            ax.plot(
                phase_days,
                phase_radii,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=label if linestyle == "-" else None,
                zorder=3,
                alpha=0.9
            )
            # ax.plot(
            #     phase_days,
            #     -phase_radii,
            #     color=color,
            #     linewidth=linewidth,
            #     linestyle=linestyle,
            #     zorder=3,
            # )

    plot_prediction_curve(
        prediction_days,
        prediction_radii_mm,
        PREDICTION_COLOR,
        "ODE prediction / predicted volume",
        1.8,
    )
    comparison_predictions = comparison_predictions or {}
    for name, (comparison_days, comparison_radii_mm, color) in comparison_predictions.items():
        comparison_radii_mm = np.asarray(comparison_radii_mm, dtype=float)
        plot_prediction_curve(
            comparison_days,
            comparison_radii_mm,
            color,
            f"{name} prediction",
            1.5,
        )
    for index, (day, radius) in enumerate(zip(label_days, label_radii_mm)):
        is_interpolation = dates[index] == interpolation_date
        label_color = (
            EVALUATION_LABEL_COLOR
            if is_interpolation
            else EVALUATION_LABEL_COLOR if day >= extrapolation_start else LABEL_COLOR
        )
        ax.vlines(
            day,
            0,
            radius,
            color=label_color,
            linewidth=1.3,
            linestyle="-",
            zorder=4,
        )
        ax.scatter(
            [day],
            [radius],
            marker=".",
            s=28,
            color=label_color,
            linewidth=1.1,
            zorder=5,
        )
    interpolation_index = dates.index(interpolation_date) if interpolation_date is not None else None
    largest_radius = max(
        float(np.max(np.abs(label_radii_mm))),
        float(np.max(np.abs(prediction_radii_mm))),
        *(
            float(np.max(np.abs(radii)))
            for _, radii, _ in comparison_predictions.values()
        ),
        1e-6,
    )
    y_limit = largest_radius * 1.2
    ax.set_ylim(0, y_limit)
    ax.set_xlim(-10, end_day + 10)
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["bottom"].set_bounds(-10, end_day + 10)
    ax.xaxis.set_ticks_position("bottom")
    x_label = ax.set_xlabel(
        "Time since first scan [days]",
        bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none"},
    )
    x_label.set_zorder(10)
    ax.set_ylabel(
        r"Spherical-equivalent lesion radius [mm]  $r = \sqrt[3]{\frac{3V}{4\pi}}$          "
    )
    # ax.grid(axis="y", which="major", color="#aeb7c0", linewidth=0.7, alpha=0.55)
    # ax.grid(axis="y", which="minor", color="#d5dbe1", linewidth=0.55, alpha=0.55)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#3f4852")
    ax.spines["bottom"].set_linewidth(1.1)
    ax.axvline(
        extrapolation_start,
        color=EVALUATION_LABEL_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=1,
        zorder=2,
        ymax=0.86,
    )
    brace_width = end_day - extrapolation_start
    if brace_width > 0:
        brace_y = y_limit * 0.86
        brace_depth = y_limit * 0.035
        x_mid = (extrapolation_start + end_day) / 2
        brace_path = MplPath(
            [
                (extrapolation_start, brace_y),
                (extrapolation_start, brace_y + brace_depth),
                (extrapolation_start + brace_width * 0.08, brace_y + brace_depth),
                (extrapolation_start + brace_width * 0.17, brace_y + brace_depth),
                (extrapolation_start + brace_width * 0.30, brace_y + brace_depth),
                (x_mid - brace_width * 0.08, brace_y + brace_depth),
                (x_mid, brace_y + 2 * brace_depth),
                (x_mid + brace_width * 0.08, brace_y + brace_depth),
                (end_day - brace_width * 0.30, brace_y + brace_depth),
                (end_day - brace_width * 0.17, brace_y + brace_depth),
                (end_day - brace_width * 0.08, brace_y + brace_depth),
                (end_day, brace_y + brace_depth),
                (end_day, brace_y),
            ],
            [
                MplPath.MOVETO,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
                MplPath.CURVE4,
            ],
        )
        ax.add_patch(
            PathPatch(
                brace_path,
                fill=False,
                color=EVALUATION_LABEL_COLOR,
                linewidth=1.2,
                clip_on=False,
            )
        )
        ax.text(
            x_mid,
            brace_y + 3.1 * brace_depth,
            "Evaluation Extrapolation phase\n(180 days before last label)",
            color=EVALUATION_LABEL_COLOR,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )
    if interpolation_index is not None:
        ax.annotate(
            "Evaluation Interpolation sample",
            xy=(
                float(label_days[interpolation_index]),
                float(label_radii_mm[interpolation_index]),
            ),
            xytext=(label_days[interpolation_index], y_limit * 0.74),
            ha="center",
            va="top",
            fontsize=8,
            color=EVALUATION_LABEL_COLOR,
            arrowprops={"arrowstyle": "-", "color": EVALUATION_LABEL_COLOR, "linewidth": 1},
        )
    panel_axes = fig.axes[:len(label_slices)]
    for connector_index, (panel_index, panel_ax) in enumerate(
        zip(panel_indices, panel_axes)
    ):
        connector = ConnectionPatch(
            xyA=(0.5, 1.0),
            coordsA=panel_ax.transAxes,
            xyB=(float(label_days[panel_index]), 0.0),
            coordsB=ax.transData,
            arrowstyle="->",
            mutation_scale=15,
            shrinkA=0,
            shrinkB=2,
            connectionstyle=_OrthogonalConnection(connector_index, len(panel_axes)),
            linestyle="-",
            linewidth=1,
            color="#000000",
            alpha=0.5,
            zorder=2,
        )
        panel_ax.scatter(
            [0.5],
            [1.0],
            transform=panel_ax.transAxes,
            color="#000000",
            s=100,
            zorder=6,
            clip_on=False,
            marker="^",
        )
        fig.add_artist(connector)

    for index, (day, radius) in enumerate(zip(label_days, label_radii_mm)):
        next_day = float(label_days[index + 1]) if index + 1 < len(label_days) else end_day
        if next_day > day:
            segments = (
                (day, min(next_day, extrapolation_start), LABEL_COLOR, (0, (5,10))),
                (max(day, extrapolation_start), next_day, EVALUATION_LABEL_COLOR, (0, (5,10))),
            )
            for segment_start, segment_end, color, linestyle in segments:
                if segment_end <= segment_start:
                    continue
                for mirrored_radius in (-radius, radius):
                    ax.hlines(
                        mirrored_radius,
                        segment_start,
                        segment_end,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.8,
                        alpha=0.5,
                        zorder=2,
                    )

    legend_handles = [
        Line2D([], [], color=LABEL_COLOR, marker="|", linestyle="none",
               markersize=11, label="Training samples"),
        Line2D([], [], color=EVALUATION_LABEL_COLOR, marker="|", linestyle="none",
               markersize=11, label="Evaluation samples"),
        Line2D([], [], color=PREDICTION_COLOR, linestyle="-.", linewidth=1.2,
               label="Model prediction during extrapolation"),
        Line2D([], [], color=PREDICTION_COLOR, linewidth=1.5, label="Neural ODE prediction"),
    ]
    legend_handles.extend(
        Line2D([], [], color=color, linewidth=1.5, label=f"{name} prediction")
        for name, (_, _, color) in comparison_predictions.items()
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"Lesion growth prediction over time of Lesion {trajectory.label_id} of Patient {trajectory.patient_id}",
        y=0.985,
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def _render_trajectory(model, dataloader, trajectory, args, device,
                       comparison_models=None, cache_signature=None):
    output_path = args.output_dir / (
        f"{trajectory.patient_id}_lesion_{trajectory.label_id}.png"
    )
    interpolation_date = _interpolation_date(trajectory)
    dates = list(trajectory.allowed_dates)
    (
        labels,
        label_days,
        affine,
        geometry,
        shape,
        lesion_z,
        label_radii_mm,
    ) = _trajectory_data(trajectory)
    if len(dates) != len(labels):
        raise ValueError("Label dates and loaded label volumes do not match.")

    label_volumes_mm3 = _volume_from_radius_mm(label_radii_mm)
    interpolation_index = dates.index(interpolation_date) if interpolation_date is not None else None
    selected_indices = _representative_label_indices(
        label_volumes_mm3,
        interpolation_index,
    )
    prediction_days = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.0,
                    float(label_days[-1]),
                    args.time_steps,
                ),
                label_days,
                [max(0.0, float(label_days[-1]) - 180.0)],
            ]
        )
    )
    patient_idx = torch.tensor([trajectory.embedding_id], dtype=torch.long, device=device)
    with torch.inference_mode():
        states = _ode_states(model, patient_idx, prediction_days)

        volume_xyz, sample_weight_mm3 = _spatial_grid(
            shape,
            affine,
            geometry,
            args.prediction_spacing_mm,
        )
        volume_encoding = _encode_chunks(model, volume_xyz, device, args.batch_size)
        prediction_volumes_mm3 = _predicted_volumes_mm3(
            model,
            volume_encoding,
            states,
            sample_weight_mm3,
            model_xyz=volume_xyz,
            batch_size=args.batch_size,
        )

        slice_xyz, slice_shape = _voxel_model_grid(shape, affine, geometry, lesion_z)
        slice_encoding = _encode_chunks(model, slice_xyz, device, args.batch_size)
        state_by_day = {float(day): state for day, state in zip(prediction_days, states)}
        prediction_slices = []
        for day in label_days:
            state = state_by_day[float(day)]
            if isinstance(state, tuple):
                slice_encoding_for_state = _encode_chunks(
                    model,
                    slice_xyz,
                    device,
                    args.batch_size,
                    state[1],
                )
            else:
                slice_encoding_for_state = slice_encoding
            mask = _prediction_slice_masks(
                model,
                slice_encoding_for_state,
                state,
            ).reshape(slice_shape)
            if mask.shape != tuple(shape[:2]):
                mask = _resize_2d(mask.astype(np.uint8), shape[:2], order=0).astype(bool)
            prediction_slices.append(mask)

    label_slices = [labels[index][:, :, lesion_z] for index in selected_indices]
    mri_slices = _aligned_mri_slices(
        dataloader,
        trajectory,
        affine,
        shape,
        lesion_z,
    )
    mri_slices = [mri_slices[index] for index in selected_indices]
    prediction_slices = [prediction_slices[index] for index in selected_indices]
    prediction_radii_mm = _radius_from_volume_mm3(prediction_volumes_mm3)
    comparison_predictions = {}
    for name, comparison_model, color in comparison_models or []:
        if isinstance(comparison_model, LesionINR):
            comparison_volumes_mm3 = _inr_prediction_volumes_mm3(
                comparison_model,
                patient_idx,
                volume_xyz,
                prediction_days,
                sample_weight_mm3,
                args.batch_size,
            )
        else:
            comparison_volumes_mm3 = _lstm_prediction_volumes_mm3(
                comparison_model,
                trajectory,
                prediction_days,
            )
        comparison_predictions[name] = (
            prediction_days,
            _radius_from_volume_mm3(comparison_volumes_mm3),
            color,
        )
    plot_data = {
        "label_slices": np.asarray(label_slices, dtype=bool),
        "label_days": label_days,
        "label_radii_mm": label_radii_mm,
        "prediction_days": prediction_days,
        "prediction_radii_mm": prediction_radii_mm,
        "mri_slices": np.asarray(mri_slices),
        "prediction_slices": np.asarray(prediction_slices, dtype=bool),
        "dates": dates,
        "panel_indices": selected_indices,
        "interpolation_date": interpolation_date,
        "comparison_predictions": comparison_predictions,
    }
    if cache_signature is not None:
        _write_plot_cache(output_path, cache_signature, plot_data)
    _render_plot_data(trajectory, plot_data, args, output_path)
    print(
        f"Plotted all {len(labels)} label timepoints for {trajectory.patient_id}, "
        f"lesion {trajectory.label_id}; displayed {len(selected_indices)} "
        f"representative MRI timepoints."
    )


def _load_comparison_models(args, device):
    models = []
    if args.no_comparisons:
        return models

    inr_uri = f"runs:/{args.inr_run_id}/{args.inr_model_artifact}"
    inr_model = load_model(inr_uri, map_location=device)
    if not isinstance(inr_model, LesionINR):
        raise TypeError(
            f"Expected LesionINR from {inr_uri}, got {type(inr_model).__name__}."
        )
    models.append(("INR", inr_model.to(device).eval(), INR_COLOR))

    lstm_uri = f"runs:/{args.lstm_run_id}/{args.lstm_model_artifact}"
    lstm_model = load_model(lstm_uri, map_location=device)
    if not isinstance(lstm_model, (LesionLSTM, LesionLatentLSTM)):
        raise TypeError(
            f"Expected LesionLSTM or LesionLatentLSTM from {lstm_uri}, "
            f"got {type(lstm_model).__name__}."
        )
    models.append(("LSTM", lstm_model.to(device).eval(), LSTM_COLOR))
    return models


def main():
    args = _arguments()
    mlflow.set_tracking_uri(args.tracking_uri)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, plotting_dataset = _plotting_dataset()
    if len(plotting_dataset) == 0:
        raise ValueError("Plotting_LesionDataset contains no samples.")
    if args.sample:
        sample_stem = Path(args.sample).stem
        matching_trajectories = [
            trajectory
            for trajectory in plotting_dataset.trajectories
            if f"{trajectory.patient_id}_lesion_{trajectory.label_id}" == sample_stem
        ]
        if not matching_trajectories:
            raise ValueError(
                f"Sample {args.sample!r} is not present in the plotting dataset."
            )
        plotting_dataset.trajectories = matching_trajectories

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_to_render = []
    cached_trajectories = []
    for trajectory in plotting_dataset.trajectories:
        output_path = args.output_dir / (
            f"{trajectory.patient_id}_lesion_{trajectory.label_id}.png"
        )
        cache_signature = _plot_cache_signature(args, trajectory)
        cached_plot_data = _read_plot_cache(output_path, cache_signature)
        if cached_plot_data is not None:
            cached_trajectories.append((trajectory, output_path, cached_plot_data))
        else:
            trajectories_to_render.append((trajectory, cache_signature))

    for trajectory, output_path, cached_plot_data in cached_trajectories:
        print(f"Rendering from cached inference data: {output_path}")
        _render_plot_data(trajectory, cached_plot_data, args, output_path)

    if not trajectories_to_render:
        print("All requested inference data was cached; no models were loaded.")
        return

    model_uri = f"runs:/{args.run_id}/{args.model_artifact}"
    model = load_model(model_uri, map_location=device)
    if not isinstance(model, (NeuralODE_INR, NeuralODETranslationINR)):
        raise TypeError(
            f"Expected NeuralODE_INR or NeuralODETranslationINR from {model_uri}, "
            f"got {type(model).__name__}."
        )
    model.to(device).eval()
    comparison_models = _load_comparison_models(args, device)

    plotting_dataset.trajectories = [trajectory for trajectory, _ in trajectories_to_render]
    print(f"Generating plots for {len(trajectories_to_render)} uncached sample(s).")
    for index, (_, cache_signature) in enumerate(trajectories_to_render):
        trajectory = plotting_dataset[index]
        _render_trajectory(
            model,
            dataloader,
            trajectory,
            args,
            device,
            comparison_models,
            cache_signature,
        )


if __name__ == "__main__":
    main()