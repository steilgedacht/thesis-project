import torch

class Config:
    """ Tracking parameters """
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow_experiment_name = "Lesion_Dialation"
    mlflow_run_name = "Naive_Dialation_Model" 


    """ Dataloading parameters """
    batchsize = 1
    num_workers = 6
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True

    lesion_trajectories_with_more_than_n_scans = 6
    use_only_growing_lesions = True

    # with that the same lesions can be in the same training batch
    training_dataset_samples_duplication_factor = 100

    dialation_iterations = 1
    background_samples_proportion = 1

    max_t = 3650.0


    """ Training parameters """
    device = "cpu"
    epochs = 1
    background_samples = 1000
    lr = 1e-4
    weight_decay = 1e-6
    scheduler_eta_min = lr * 0.0001
    max_grad_norm_clip = 1.0
    only_train = False

    from utils.loss_bce_dice import Loss_BCE_Dice
    loss_fn = Loss_BCE_Dice()


    """ Model parameters """
    from utils.model_dialation import Dilation_Model
    model = Dilation_Model
    model_params = {}
    model_save_name = "lesion_dialation_model"


    """ Logging parameters """
    train_log_interval = 10
    train_delete_cache_interval = 10
    n_monitoring_samples_to_visualize = 5
    validation_interval = 20
    print_loss_interval = 10
    
    time_evolution_steps = 100
    time_evolution_side_length = 50

    full_size_side_length = 500