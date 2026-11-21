"""Plot a saved patient-wise model from an MLflow run.

Run from any directory:
    python experiments/12_patient-specific/plot_past_run.py

The generated artifacts are logged to a new MLflow run in the source run's
experiment; the source training run is left unchanged.
"""

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient

import train_ode as lesion_training
from utils import model_neural_ode_translation_validation
from utils import model_neural_ode_validation
from utils.mri_dataloader import MRI_Dataloader


DEFAULT_RUN_ID = "e8566f5eb050488d996cda41af8251b9"


def load_patient_plotting():
    plotting_path = ROOT / "experiments" / "12_patient-specific" / "patient_plotting.py"
    spec = importlib.util.spec_from_file_location(
        "patient_specific_plotting", plotting_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load patient plotting module from {plotting_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_param(params, name, default):
    value = params.get(name)
    if value is None:
        return default
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_model(model_uri, device):
    compatibility_modules = {
        "utils.model_neural_ode": model_neural_ode_validation,
        "utils.model_neural_ode_translation": (
            model_neural_ode_translation_validation
        ),
    }
    original_modules = {
        name: sys.modules.get(name) for name in compatibility_modules
    }
    sys.modules.update(compatibility_modules)
    try:
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
    finally:
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--config",
        default="configs/06_neural_ode/config_patient_wise.py",
        help="Config used to locate MLflow and set plotting options.",
    )
    parser.add_argument("--tracking-uri", help="Override the config's MLflow URI.")
    parser.add_argument("--model-artifact", help="Override the model artifact name.")
    parser.add_argument("--patient-id", help="Plot only this patient.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (default: CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "--side-length",
        type=int,
        default=96,
        help="Number of query points per side for each plotted slice.",
    )
    args = parser.parse_args()
    if args.side_length < 2:
        parser.error("--side-length must be at least 2.")

    config = lesion_training.add_base_configurations(
        lesion_training.load_config(args.config)
    )
    mlflow.set_tracking_uri(args.tracking_uri or config.mlflow_tracking_uri)
    client = MlflowClient()
    source_run = client.get_run(args.run_id)
    params = source_run.data.params

    model_artifact = (
        args.model_artifact
        or params.get("model_save_name")
        or config.model_save_name
    )
    model_uri = f"runs:/{args.run_id}/{model_artifact}"
    device = torch.device(args.device)
    print(f"Loading {model_uri} on {device}.")
    model = _load_model(model_uri, device)

    min_scans = int(
        _run_param(
            params,
            "lesion_trajectories_with_more_than_n_scans",
            config.lesion_trajectories_with_more_than_n_scans,
        )
    )
    only_growing = bool(
        _run_param(params, "use_only_growing_lesions", config.use_only_growing_lesions)
    )
    mri_dataloader = MRI_Dataloader()
    mri_dataloader.cache_lesion_trajectories_from_n_scans(
        n_scans=min_scans,
        only_growing=only_growing,
    )
    trajectories = mri_dataloader.cache_lesion_trajectories
    if trajectories is None:
        raise RuntimeError("No lesion trajectories were loaded.")

    patient_plotting = load_patient_plotting()
    plotting_dataset = patient_plotting.PatientPlottingDataset(trajectories)
    if args.patient_id:
        plotting_dataset.samples = [
            sample
            for sample in plotting_dataset.samples
            if sample.patient_id == args.patient_id
        ]
        if not plotting_dataset.samples:
            raise ValueError(
                f"Patient {args.patient_id!r} is not in the plotting sample set."
            )
    if not len(plotting_dataset):
        raise ValueError("The plotting sample set contains no selected patients.")

    config.patient_plot_side_length = args.side_length
    with mlflow.start_run(
        experiment_id=source_run.info.experiment_id,
        run_name=f"patient_plots_{args.run_id[:8]}",
        tags={"source_run_id": args.run_id, "run_type": "patient-wise plotting"},
    ):
        mlflow.log_params(
            {
                "source_run_id": args.run_id,
                "model_uri": model_uri,
                "patient_count": len(plotting_dataset),
                "plot_side_length": args.side_length,
            }
        )
        for sample in plotting_dataset:
            print(f"Plotting patient {sample.patient_id}.")
            patient_plotting.plot_patient_contact_sheet(model, 0, sample, config)
            patient_plotting.plot_patient_time_evolution(model, 0, sample, config)


if __name__ == "__main__":
    main()
