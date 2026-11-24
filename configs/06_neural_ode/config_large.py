"""Neural ODE (translation), 2x network size, trained from scratch for 200 epochs.

For a 48 GB GPU server. Based on config.py (server defaults) with every fix of
config_fixes.py, but NO warm start and NO 6 GB settings.

Size: the "parameters" of the 6 GB model are 72.6M, but 68.4M of that is the
patient_embedding table (133,502 rows indexed by the global lesion id, only
238 of them trained). The network itself is 4.26M; this config doubles it:

    6 GB model  : decoder width 256, ODE width 512   -> network 4.26M
    this config : decoder width 512, ODE width 724   -> network ~8.3M (1.95x)

Memory (measured, one real training step incl. TV-time double backward,
10k points per item): ~1.15 GiB fixed + ~0.61 GiB per item
    batch 16 -> ~11 GiB, batch 32 -> ~21 GiB.

Batch 16 / lr 1e-4: 298 steps per epoch, ~60k optimizer steps in 200 epochs,
about what the best 6 GB NODE got over its three runs (batch 8). lr >= 1.8e-4
kept the loss flat in run 287d9894c8ce4400bb0c32c6e46c45c8, so it stays 1e-4.
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("neural_ode_base_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "ODE large (2x), fixes, 200 epochs"

    """ Training """
    batchsize = 16
    epochs = 200
    lr = 1e-4
    scheduler_eta_min = lr * 0.0001
    warm_start_run_id = None            # from scratch

    """ Logging / validation """
    validation_interval = 5
    plot_validation_interval = 25
    # Every step has the same 16 x 10k points, so emptying the CUDA cache
    # every 10 steps only forces reallocation.
    train_delete_cache_interval = 1000

    """ Sampling fixes (see config_fixes.py) """
    oversample_positive_samples = False
    voxel_jitter = True                 # points inside the voxel, not only on slice planes
    single_time_items = True            # one time per item (the NODE uses the mean time)
    near_field_fraction = 0.5           # half of the negatives from the lesion box + 20 mm
    near_field_margin_mm = 20.0

    """ Model: 2x network """
    model_params = {
        **_base.Config.model_params,
        "spatial_hidden_dim": 512,      # 6 GB model: 256
        "ode_hidden_dim": 724,          # 6 GB model / config.py: 512
        "use_region_prior": True,
        "region_margin_mm": 10.0,
        "region_strength_per_mm": 1.0,
        "region_prior_in_training": False,  # prior only in validation / plotting / arena
    }
