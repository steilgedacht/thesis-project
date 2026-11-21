import torch


def _mlflow_uri():
    try:
        host = open('/home/optima/bbergmann/mlflow_host.txt').read().strip()
        return f"http://{host}:5124"
    except OSError:
        return "http://127.0.0.1:5124"


class Config:
    """ Tracking parameters """
    mlflow_tracking_uri = _mlflow_uri()
    mlflow_experiment_name = "Lesion_Point_LSTM_Training"
    mlflow_run_name = "Point-LSTM"
    seed = 0


    """ Dataloading parameters """
    batchsize = 32
    validation_batchsize = 8
    num_workers = 7
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True
    preload_processes = 8          # parallel NIfTI -> ScanEntry conversion at startup

    lesion_trajectories_with_more_than_n_scans = 6
    use_only_growing_lesions = True
    training_dataset_samples_duplication_factor = 20

    # history_points / max_history_len are taken from model_params
    dataset_params = dict(
        query_points=3000,
        consecutive_history_probability=0.5,   # else skip scans -> longer forecast gaps
        ring_mm=3.0,                           # boundary band for history/query sampling
    )


    """ Training parameters """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    lr = 4e-4
    weight_decay = 1e-6
    scheduler_eta_min = lr * 0.01
    max_grad_norm_clip = 1.0
    only_train = False

    from utils.loss_node import Loss_BCE_Dice_TV
    # Sampling is already roughly balanced; TV is off (no coords passed).
    loss_fn = Loss_BCE_Dice_TV(lambda_space=0.0, lambda_time=0.0, use_dynamic_pos_weight=False)


    """ Model parameters """
    from utils.model_point_lstm import PointLSTM
    model = PointLSTM
    model_params = dict(
        num_fourier_frequencies=8,
        encoder_hidden_dim=256,
        encoder_layers=3,
        scan_code_dim=256,
        lstm_hidden_dim=256,
        lstm_layers=2,
        latent_dim=256,
        head_hidden_dim=256,
        spatial_hidden_dim=256,
        spatial_layers=8,
        time_frequencies=4,
        max_t=3650.0,
        history_points=1024,
        max_history_len=5,
        use_sdf_prior=False,
        sdf_clip_mm=20.0,
        use_lesion_embedding=False,
        lesion_embed_dim=32,
        lesion_embedding_dropout=0.5,
    )
    model_save_name = "point_lstm_model"


    """ Logging parameters """
    train_log_interval = 10
    validation_interval = 5
    plot_validation_interval = 25
    n_monitoring_samples_to_visualize = 5
