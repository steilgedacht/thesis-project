import torch

class Config:
    """ Tracking parameters """
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow_experiment_name = "Lesion_INR_MetaLearning"
    mlflow_run_name = "MAML_INR" 


    """ Dataloading parameters """
    batchsize = 100
    num_workers = 7
    prefetch_factor = 2
    pin_memory = True
    persistent_workers = True

    lesion_trajectories_with_more_than_n_scans = 6
    use_only_growing_lesions = True

    # with that the same lesions can be in the same training batch
    training_dataset_samples_duplication_factor = 20 

    dialation_iterations = 1
    background_samples_proportion = 1


    """ Training parameters """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 200
    background_samples = 3000
    lr = 1e-4
    weight_decay = 1e-6
    scheduler_eta_min = lr * 0.0001
    max_grad_norm_clip = 1.0
    only_train = False

    from utils.loss_bce_dice_tv_meta import Loss_BCE_Dice_TV
    
    # Dynamic class weighting based on ground truth imbalance
    # Automatically adapts pos_weight per batch from label class ratio
    use_dynamic_pos_weight = True
    
    # Total Variation regularization for spatial smoothness
    use_total_variation_loss = True
    tv_loss_weight = 0.01
    
    # Instantiate loss with dynamic weighting
    loss_fn = Loss_BCE_Dice_TV(
        lambda_space=2e-2,
        lambda_time=0.6,
        use_dynamic_pos_weight=use_dynamic_pos_weight,
    )

    """ Meta-Learning specific parameters """
    # Inner loop (patient adaptation) hyperparameters
    inner_lr = 0.01                # Learning rate for inner loop SGD
    inner_steps = 3                # Number of inner loop update steps per task

    """ Model parameters """
    from utils.model_inr_meta import LesionINR
    model = LesionINR
    model_params = {
        "latent_dim" : 128,
        "input_dim" : 4,
        "hidden_dim" : 512,
        "output_dim" : 1,
        "omega_0" : 30.0,
        "n_layers" : 8,
        "time_freqs" : 6,
        "max_t" : 3650.0,
    }
    model_save_name = "lesion_inr_meta_learning"


    """ Logging parameters """
    train_log_interval = 10
    train_delete_cache_interval = 10
    n_monitoring_samples_to_visualize = 5
    validation_interval = 50
    print_loss_interval = 10
    
    time_evolution_steps = 100
    time_evolution_side_length = 50

    full_size_side_length = 500
