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
    loss_fn = Loss_BCE_Dice()
    use_total_variation_loss = True


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
    dataset_params = {
        "grid_size": lstm_grid_size,
        "margin_mm": 20.0,
        "min_extent_mm": 80.0,
        "max_sequence_len" : 6,
        "device": device
    }

    


    """ Model parameters """
    from utils.model_lstm import LesionLSTM
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