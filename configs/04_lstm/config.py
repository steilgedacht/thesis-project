import torch

class Config:
    """ Tracking parameters """
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow_experiment_name = "Lesion_INR_Training"
    mlflow_run_name = "LSTM" 


    """ Training parameters """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 201
    background_samples = 1500
    lr = 1e-4
    weight_decay = 1e-6
    scheduler_eta_min = lr * 0.0001
    max_grad_norm_clip = 1.0
    only_train = False

    from utils.loss_bce_dice import Loss_BCE_Dice

    # Dynamic class weighting based on ground truth lesion proportion
    # The loss will compute pos_weight = (negative_pixels / positive_pixels)
    # for each batch, automatically adapting to varying lesion sizes
    use_dynamic_pos_weight = True

    # Total Variation loss weight for spatial smoothness
    # Encourages neighboring voxels to have similar predictions
    # Reduces noise and promotes coherent lesion regions
    use_total_variation_loss = True
    tv_loss_weight = 0.01  # Scale relative to BCE+Dice (try 0.001-0.05)

    # Instantiate loss with dynamic weighting
    loss_fn = Loss_BCE_Dice(
        use_dynamic_pos_weight=use_dynamic_pos_weight,
        use_tv_loss=use_total_variation_loss,
        tv_weight=tv_loss_weight
    )


    """ Dataloading parameters """
    batchsize = 1
    gradient_accumulation_steps = 128
    num_workers = 7
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True

    lesion_trajectories_with_more_than_n_scans = 4
    use_only_growing_lesions = True

    # with that the same lesions can be in the same training batch
    training_dataset_samples_duplication_factor = 20 

    dialation_iterations = 1
    background_samples_proportion = 1

    lstm_grid_size = (64, 64, 64)
    max_sequence_len = 6  # Max observation history for plotting
    dataset_params = {
        "grid_size": lstm_grid_size,
        "margin_mm": 20.0,
        "min_extent_mm": 80.0,
        "max_sequence_len" : max_sequence_len,
        "device": device
    }

    """ Model parameters """
    from utils.model_lstm import LesionLSTM, LesionLatentLSTM

    # If True, use an autoencoder to encode each visit into a latent vector and
    # run an RNN in latent space (encoder + latent-LSTM + decoder). If False,
    # use the ConvLSTM that operates on full grids.
    use_autoencoder_lstm = True


    if use_autoencoder_lstm:
        model = LesionLatentLSTM
        model_params = {
            "latent_dim": 256,
            "hidden_dim": 256,
            "n_layers": 2,
            "time_freqs": 6,
            "max_t": 3650.0,
            # encoder/decoder params can be tuned; keep sensible defaults
            "encoder_params": {"base_channels": 32},
            "decoder_params": {"base_channels": 32, "out_size": lstm_grid_size},
            "use_patient_embedding": False,
        }
        model_save_name = "lesion_lstm_autoencoder"
    else:
        model = LesionLSTM
        model_params = {
            "grid_size" : lstm_grid_size,
            "latent_dim" : 128,
            "hidden_channels" : 32,
            "n_layers" : 2,
            "time_freqs" : 6,
            "max_t" : 3650.0,
            "kernel_size" : 3
        }
        model_save_name = "lesion_lstm"

    """ Logging parameters """
    train_log_interval = 10
    train_delete_cache_interval = 10
    n_monitoring_samples_to_visualize = 5
    validation_interval = 10
    print_loss_interval = 10
    
    time_evolution_steps = 100
    time_evolution_side_length = 100

    full_size_side_length = 500
