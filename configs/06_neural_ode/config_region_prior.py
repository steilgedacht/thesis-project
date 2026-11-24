"""Neural ODE (translation variant) with the region prior, sized for a 6 GB GPU.

Region prior: points more than ``region_margin_mm`` outside the box around a
lesion's training masks get ``region_strength_per_mm`` logits per mm
subtracted, which suppresses spurious far-away blobs (utils/region_prior.py).

Local 6 GB settings (measured with one real training step, TV loss on,
10k points per sample; peak = torch max_memory_allocated):

    config.py as is (width 512, batch 16)   -> out of memory
    width 512, batch 4                      -> 3.4 GiB, 426 ms/step, ~14 h for 100 epochs
    width 256, batch 8  (this config)       -> 3.4 GiB, 343 ms/step, ~1.7 h for 30 epochs

Width 256 x 8 layers is the same decoder size as the Point-LSTM, and 30 epochs
at batch 8 are ~18k optimizer steps, about what the Point-LSTM gets (100 epochs
at batch 32). Everything else is inherited from config.py.
"""
import importlib.util
from pathlib import Path

_base_path = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("neural_ode_base_config", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Config(_base.Config):
    mlflow_run_name = "ODE + region prior (6 GB)"

    """ Memory / runtime for a 6 GB GPU """
    batchsize = 8                       # config.py: 16
    epochs = 50                         # config.py: 100 (~18k steps at batch 8)
    lr = 1e-4
    scheduler_eta_min = _base.Config.lr * 0.0001
    validation_interval = 5
    plot_validation_interval = 15       # config.py: 25 (would only plot once in 30 epochs)
    validation_grid_chunk_size = 65536  # config.py: 262144, smaller inference chunks
    # config.py: 10. Every step has the same 8 x 10k points, so emptying the
    # CUDA cache only forces reallocation (GPU dips to ~20%). The step counter
    # restarts each epoch (~595 steps), so this empties it once per epoch start.
    train_delete_cache_interval = 1000

    model_params = {
        **_base.Config.model_params,
        "spatial_hidden_dim": 256,      # config.py: 512, same width as the Point-LSTM decoder
        "use_region_prior": True,
        "region_margin_mm": 10.0,
        "region_strength_per_mm": 1.0,
    }
