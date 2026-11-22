"""INR with the region prior; everything else as config.py.

Points more than ``region_margin_mm`` outside the box around a lesion's
training masks get ``region_strength_per_mm`` logits per mm subtracted, which
suppresses the spurious far-away blobs. See utils/region_prior.py.
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("inr_base_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


def _mlflow_uri():
    try:
        host = open('/home/optima/bbergmann/mlflow_host.txt').read().strip()
        return f"http://{host}:5124"
    except OSError:
        return "http://127.0.0.1:5124"


class Config(_base.Config):
    mlflow_tracking_uri = _mlflow_uri()
    mlflow_run_name = "INR + region prior"
    model_params = {
        **_base.Config.model_params,
        "use_region_prior": True,
        "region_margin_mm": 10.0,
        "region_strength_per_mm": 1.0,
    }
