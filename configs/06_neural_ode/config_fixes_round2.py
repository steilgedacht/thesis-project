"""Second training round of config_fixes.py, warm-started from its finished run.

Round 1 (b590863666a74846a6a20c25a5ef35d0, itself warm-started from
c67473bde7ba44c1959d2cbce158397b) still improved at every validation until the
cosine schedule had decayed the lr to ~0:

    extrapolation Dice 0.54 -> 0.59 -> 0.62 -> 0.64 -> 0.65 -> 0.66
    interpolation Dice 0.45 -> 0.58 -> 0.66 -> 0.68 -> 0.72 -> 0.71

This round restarts the same schedule (lr 1e-4, 50 epochs, cosine) from those
weights. Everything else is config_fixes.py.
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config_fixes.py")
_spec = importlib.util.spec_from_file_location("neural_ode_fixes_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "ODE + fixes, round 2 (6 GB)"
    warm_start_run_id = "b590863666a74846a6a20c25a5ef35d0"
