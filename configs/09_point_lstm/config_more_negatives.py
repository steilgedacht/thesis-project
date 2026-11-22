"""Point-LSTM with more query points away from the lesion boundary.

The default sampler puts ~75% of the query points on the lesion, its 3 mm ring
or the ring of the last history scan, so regions a few cm away are almost never
supervised and the decoder produces spurious blobs there. This config shifts
half of the points to the box around the lesion and the whole volume.
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("point_lstm_base_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "Point-LSTM + more negatives"
    extra_config_files = [str(_base_path)]
    # (lesion, target ring, last-history ring, box around lesions, whole volume)
    # default:  (0.30, 0.30, 0.15, 0.10, 0.15)
    dataset_params = {**_base.Config.dataset_params,
                      "query_fractions": (0.20, 0.20, 0.10, 0.30, 0.20)}
