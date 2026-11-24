"""INR with the region prior, sized for a 6 GB GPU; everything else as config.py.

Region prior: points more than ``region_margin_mm`` outside the box around a
lesion's training masks get ``region_strength_per_mm`` logits per mm
subtracted, which suppresses spurious far-away blobs (utils/region_prior.py).

Local 6 GB settings (measured with real training steps, TV loss on,
1500 points per sample; peak = torch max_memory_allocated; ~0.9 GiB of the
card is taken by the desktop):

    config.py batch 200 (300k points/step)        -> out of memory
    old embedding table (476k rows), batch 16     -> 3.8 GiB, 331 ms/step
    compact table (one row per lesion), batch 16  -> 3.1 GiB, 313 ms/step
                                        batch 20  -> 3.9 GiB, 336 ms/step  (this config)
                                        batch 22  -> 4.3 GiB, 345 ms/step
                                        batch 24  -> out of memory (4.7 GiB and 447 ms/step
                                                     with expandable_segments)
    without TV loss (no double backward)  batch 20  -> 1.7 GiB, 151 ms/step
                                          batch 48  -> 4.0 GiB, 317 ms/step  (this config)
                                          batch 64  -> out of memory

The SIREN keeps width 512 x 8 layers. The embedding table now has one row per
training lesion (238) instead of len(trajectories) * 100 = 476k rows; the
global embedding ids are mapped to rows inside the model (utils/model_inr.py).
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
    mlflow_run_name = "INR + region prior (6 GB)"

    """ Memory / runtime for a 6 GB GPU """
    batchsize = 48                      # config.py: 200 (out of memory); 64 does not fit (no TV, see below)
    epochs = 200                        # config.py: 151; ~99 steps/epoch -> ~20k steps, ~1.75 h + validation
    validation_interval = 10            # config.py: 5
    # lr stays 1e-4 (config.py); the NODE only started learning below ~1.5e-4
    validation_grid_chunk_size = 65536  # smaller inference chunks for grid validation
    # config.py: 10. Every step has the same 48 x 1500 points, so emptying the
    # CUDA cache only forces reallocation. The step counter restarts each
    # epoch (~99 steps), so this empties it once per epoch start.
    train_delete_cache_interval = 1000

    """ Sampling (first local run: Dice 0.10, flat from epoch 5, ~66 cm3 false volume per case) """
    # train_inr.py used to ignore this and fall back to the dataset default
    # True: half of each item's observed points were (repeated) lesion voxels.
    oversample_positive_samples = False
    # Half of the background negatives come from the lesion box grown by
    # 20 mm instead of the whole volume. Before: ~48 of 1500 points per item
    # in the 0-20 mm shell around the lesion, now ~325. That shell is where the
    # false-positive halo sat (the region prior only acts beyond box + 10 mm).
    near_field_fraction = 0.5
    near_field_margin_mm = 20.0
    # Second run (with the two fixes above): Dice +1%, still ~60 cm3 false volume,
    # flat from epoch 5. The scans have 5-6.5 mm slices, so without jitter every
    # training point lies on a slice plane, while the grid validation queries
    # z every 1.34 mm; ~3/4 of it lies between slices, where the SIREN was
    # never supervised. Jitter samples the whole voxel (as the Point-LSTM does).
    voxel_jitter = True
    # The INR uses the time of point 0 for the whole item, but items mixed scan
    # points with random intermediate times: 35% of items trained every point
    # at a wrong time (mean error 108 days). One time per item fixes that.
    single_time_items = True

    """ Collapse fix (third run: raw logit 1.10 +- 0.008 over the whole volume) """
    # The epoch-50 checkpoint of the third run output a constant: the FiLM
    # gammas had shrunk from 1.0 to ~0.1, so position no longer reached the
    # output, and the region prior alone cut the "lesion everywhere" output
    # into box + 10 mm (hence ~60 cm3 false volume in every run). Short CPU
    # trainings (20 lesions, 600 steps) on fresh training points:
    #
    #   prior in training  TV 0.03   near-field FP 0.85  positives 0.84  spatial std 0.011  gamma 0.64
    #   prior off          TV 0.03   near-field FP 0.54  positives 0.58  spatial std 0.009  gamma 0.57
    #   prior in training  TV 0      near-field FP 0.85  positives 0.85  spatial std 0.10   gamma 0.95
    #   prior off          TV 0      near-field FP 0.16  positives 0.71  spatial std 0.59   gamma 0.91
    #
    # TV flattens the SIREN (gammas shrink) and the prior during training makes
    # "positive near the lesion" a cheap answer. So: no TV, and the prior is
    # only applied in eval mode (validation, plotting, arena).
    use_total_variation_loss = False

    model_params = {
        **_base.Config.model_params,
        "region_prior_in_training": False,
        "use_region_prior": True,
        "region_margin_mm": 10.0,
        "region_strength_per_mm": 1.0,
    }
