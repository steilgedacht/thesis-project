"""NODE (translation, region prior, 6 GB) with the dataset / prior fixes found for the INR.

Diagnosis of run c67473bde7ba44c1959d2cbce158397b (config_region_prior.py,
validation Dice 0.585 extrapolation / 0.638 interpolation), uniform 1.34 mm grid:

    * On its own TRAINING scans the grid Dice is only 0.65, although the
      training loss reports Dice ~0.98 on the sampled points. The model is
      right on the training points and wrong in between: with 5-6.5 mm
      slices every training point sits on a slice plane, and the 0-10 mm band
      around the lesion is barely sampled.
    * It over-predicts: 32 of 42 extrapolation cases too large (median 1.4x),
      interpolation 2.6 cm3 predicted vs 1.3 cm3 target, almost no FN.
    * Without the region prior it puts ~37 cm3 of blobs >10 mm away, i.e. it
      relies on the prior like the INR did.
    * Each item mixes scan-time points with random intermediate times, and
      the NODE uses the mean time: 68 days off at the median.

Fixes (same as configs/02_tv_regulation/config_region_prior.py, where they
took the INR from Dice 0.11 to 0.52):
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config_region_prior.py")
_spec = importlib.util.spec_from_file_location("neural_ode_region_prior_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "ODE + fixes (6 GB)"

    # Every point inside its voxel, not only on the slice planes.
    voxel_jitter = True
    # One time per item (the NODE uses the mean time of the item).
    single_time_items = True
    # Half of the background negatives from the lesion box grown by 20 mm.
    near_field_fraction = 0.5
    near_field_margin_mm = 20.0

    # Start from the weights of the finished run (None = train from scratch).
    # Saves most of the ~3 h; for a clean comparison in the thesis, train
    # from scratch instead.
    warm_start_run_id = "c67473bde7ba44c1959d2cbce158397b"

    model_params = {
        **_base.Config.model_params,
        # Prior only in validation / plotting / arena, not in the training loss.
        "region_prior_in_training": False,
    }
