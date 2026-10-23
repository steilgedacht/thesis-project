import torch


class Config:
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow_experiment_name = "Lesion_Mamba_Training"
    mlflow_run_name = "Mamba"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 101
    lr = 1e-4
    weight_decay = 1e-6
    scheduler_eta_min = lr * 0.0001
    max_grad_norm_clip = 1.0
    only_train = False

    autoencoder_pretrain_epochs = 10
    mamba_only_epochs = 10
    load_autoencoder_weights_path = None
    save_autoencoder_weights_path = "mamba_autoencoder_weights.pth"
    freeze_loaded_autoencoder = True

    from utils.loss_bce_dice import Loss_BCE_Dice

    loss_fn = Loss_BCE_Dice(
        use_dynamic_pos_weight=True,
        use_tv_loss=False,
        tv_weight=0.01,
        edge_weight=4.0,
    )

    batchsize = 1
    gradient_accumulation_steps = 128
    num_workers = 7
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True

    lesion_trajectories_with_more_than_n_scans = 4
    use_only_growing_lesions = True
    training_dataset_samples_duplication_factor = 20

    mamba_grid_size = (64, 64, 64)
    lstm_grid_size = mamba_grid_size
    max_sequence_len = 6
    dataset_params = {
        "grid_size": mamba_grid_size,
        "margin_mm": 20.0,
        "min_extent_mm": 80.0,
        "max_sequence_len": max_sequence_len,
        "device": device,
    }

    from utils.model_mamba import LesionLatentMamba

    model = LesionLatentMamba
    model_params = {
        "grid_size": mamba_grid_size,
        "latent_dim": 256,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "n_layers": 2,
        "time_freqs": 6,
        "max_t": 3650.0,
        "encoder_params": {"base_channels": 32},
        "decoder_params": {"base_channels": 32},
        "use_patient_embedding": True,
    }
    model_save_name = "lesion_latent_mamba"

    train_log_interval = 10
    train_delete_cache_interval = 10
    validation_interval = 10
    print_loss_interval = 10
    time_evolution_steps = 100
    time_evolution_side_length = 100
    full_size_side_length = 500
