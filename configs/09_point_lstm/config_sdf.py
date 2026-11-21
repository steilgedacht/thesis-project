"""Point-LSTM with the signed-distance prior; everything else as config.py."""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("point_lstm_base_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "Point-LSTM + SDF prior"
    extra_config_files = [str(_base_path)]
    model_params = {**_base.Config.model_params, "use_sdf_prior": True}
