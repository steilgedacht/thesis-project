"""INR, 2x network size, for a 48 GB GPU server; all fixes of config_region_prior.py.

Size (the embedding table is one 256-d row per training lesion, 238 rows):

    config_region_prior.py : width 512, latent 128  -> network 3.33M
    this config            : width 640, latent 256  -> network 6.38M (1.92x)

More points per item: 3000 instead of 1500. The best local run
(adea3d5301684f30aed0345694012672) peaked early (extrapolation 0.52,
interpolation 0.45 around epoch 30-40), dropped to ~0.46 / ~0.32 while the
training Dice loss kept falling (~0.03), and only recovered as the lr decayed:
it overfits. A larger SIREN alone would make that worse, so the extra memory
also goes into denser supervision per step, at the same batch and step count.

Memory (measured, one real training step, no TV): ~0.21 GiB per item at
3000 points -> batch 48 ~10 GiB. 99 steps per epoch, ~20k steps in 200 epochs.
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config_region_prior.py")
_spec = importlib.util.spec_from_file_location("inr_region_prior_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "INR large (2x), fixes, 200 epochs"

    batchsize = 48
    epochs = 200
    background_samples = 3000           # points per item; config_region_prior.py: 1500
    validation_interval = 5             # see the early peak; config_region_prior.py: 10
    validation_grid_chunk_size = 262144  # server default (6 GB config: 65536)

    model_params = {
        **_base.Config.model_params,
        "hidden_dim": 640,              # config_region_prior.py: 512
        "latent_dim": 256,              # config_region_prior.py: 128
    }
