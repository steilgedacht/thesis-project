import torch

class Config:
    """ Tracking parameters """
    mlflow_tracking_uri = "http://127.0.0.1:5000"
    mlflow_experiment_name = "Lesion_ODE_Training"
    mlflow_run_name = "ODE" 


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
    epochs = 300
    background_samples = 3000
    lr = 1e-4
    weight_decay = 1e-6
    scheduler_eta_min = lr * 0.0001
    max_grad_norm_clip = 1.0
    only_train = False

    from utils.loss_bce_dice_tv import Loss_BCE_Dice_TV
    
    # Dynamic class weighting based on ground truth imbalance
    # Automatically adapts pos_weight per batch from label class ratio
    use_dynamic_pos_weight = True
    
    # Total Variation regularization for spatial smoothness
    use_total_variation_loss = True
    tv_loss_weight = 0.01
    
    # Instantiate loss with dynamic weighting
    loss_fn = Loss_BCE_Dice_TV(
        lambda_space=3e-3, 
        lambda_time=0.01
    )


    """ Model parameters """
    from utils.model_neural_ode import NeuralODE_INR
    model = NeuralODE_INR
    model_params = dict(
        patient_embed_dim=32,
        latent_dim=256,
        ode_hidden_dim=256,
        ode_layers=5,
        ode_steps=16,        # mehr = genauer, aber langsamer/mehr Speicher (Backprop through time)
        ode_method="rk4",    # oder "euler"
        spatial_hidden_dim=256,
        spatial_layers=8,
        num_fourier_frequencies=12,  # 0 = deaktiviert
    )

    model_save_name = "lesion_ode_model"


    """ Logging parameters """
    train_log_interval = 10
    train_delete_cache_interval = 10
    n_monitoring_samples_to_visualize = 5
    validation_interval = 20
    print_loss_interval = 10
    
    time_evolution_steps = 100
    time_evolution_side_length = 50

    full_size_side_length = 500