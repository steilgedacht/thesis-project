# Longitudinal growth analysis of brain metastasis

This repository contains a research and training pipeline for modeling lesion evolution over time in medical imaging data.

## Installation

The 150 GB of data has to be downloaded from this [Website](https://www.cancerimagingarchive.net/collection/yale-brain-mets-longitudinal/) and saved in the following folder structure:

`data/entire_yale_dataset/PRE_POST_YBML`

The lesion segmentations have then to be in 
`data/entire_yale_dataset/predictions`.

To test out the training, only the ready-processed trajectories is needed which can be downloaded via this drive: 

TODO

As they only contain the filled segmentation masks in a binary datatype, the files can be nicely compressed and shrink down to below 200MB. Once the zip is downloaded, extract the file into the folder `data/entire_yale_dataset/predictions` and extract it there so that you have the following structue:

`data/entire_yale_dataset/predictions/YG_**`

Afterwards, create a virtual environment and install the packages listed in [requirements.txt](requirements.txt):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training Commands & Examples

### Commands to reproduce the Papers results

```bash
# Global Dilation Model
python3 train_dilation.py --config configs/01_dialation/config.py

# Patient specific dilation Model 
python3 train_dilation.py --config configs/01_dialation/config_patient_specific.py

# LSTM
python3 train_lstm.py --config configs/04_lstm/config.py

# INR
python3 train_inr.py --config configs/02_tv_regulation/config.py

# NODE
python3 train_ode.py --config configs/06_neural_ode/config.py
```

### Example 1, starting the training locally:

To train the default model:

```bash
python train_inr.py --config configs/02_tv_regulation/config.py
```

To run an LSTM experiment:

```bash
python train_lstm.py --config configs/04_lstm/config.py
```

### Example 2, submit a model to the server and train it there:

```bash
# enable vpn
echo "password" | sudo openconnect vpn.meduniwien.ac.at --authgroup="Mitarbeiter_exkl._Journale" -u username --passwd-on-stdin --background

# adjusting config
vi configs/02_tv_regulation/config.py

# syncing to the server
rsync -avP --exclude='.git' --exclude='__pycache__/' --exclude='venv/' --exclude='data' --exclude='server/mlartifacts' --exclude='mlartifacts' --exclude='mlruns' --exclude='mlflow.db' --exclude='.archive'  . nodel:~/projects/thesis

# submit a job via slurm
sbatch slurm_jobs/train.sbatch

# If you want to see how the run is going while it is still running, inspect with
sh peak_server.sh

# If you want to merge the Mlflow run from the server with the local Mlflow database:
sh sync_server.sh
```

### Example 3, features of the dataloader utils:

Load a patient and register the images
```python
from utils.patient import Patient
p = Patient("YG_QLW4GDHFNOUK")
p.register_all_to_first()
```

Visualize a MRI Scan in a animation 
```python
from utils.mri_dataloader import MRI_Dataloader
mdl = MRI_Dataloader()
sample = mdl.find_by_patient_id_and_date("YG_QLW4GDHFNOUK", "2014-12-18")
sample.plot_mri_animation()
```

Load a trajectory, plot a animation of it's samples and print the dates
```python
from utils.mri_dataloader import MRI_Dataloader
mdl = MRI_Dataloader()
trj = mdl.find_lesion_trajectory("YG_BLAPRRKW79HF",2)
trj.plot_animation()
print(trj.dates)
```

Process each sample for a Patient, register them and then extract the lesion trajectories out of it.
```python
from utils.patient import Patient
p = Patient("YG_QLW4GDHFNOUK")
p.process_samples()
p.register_all_to_first()
p.merge_lesion_to_trajectory()
```

## Repository layout

### Root-level scripts

These files are the main entry points for running experiments.

- `train.py`  
  Main training script for the default lesion INR model. It loads a config, sets up MLflow, builds the datasets, and launches training.
- `train_lstm.py`  
  Training pipeline for a temporal LSTM-based model. This includes staged training logic with autoencoder pretraining and frozen-weights phases.
- `train_meta.py`  
  Meta-learning variant using MAML-style adaptation. This is designed for patient-specific adaptation and task-based training.
- `train_autoencoder.py`  
  Training script for autoencoder-based reconstruction experiments, often used to stabilize latent representations before temporal modeling.
- `train_ode.py`  
  Neural ODE / continuous-time variant of the training pipeline.
- `train_finite_elemente.py`  
  Additional model/training entry point for finite-element-like or PDE-inspired experiments.
- `peak_server.sh`, `start_mlflow.sh`, `sync_server.sh`  
  Helper scripts for starting MLflow, syncing data between local machines and remote servers, and keeping the server environment ready.

### `experiments/`

This folder contains exploratory work and research notebooks. The number just determines the historical order in which they were created. Each time something has to be researched, a new folder is created.
- `01_dataset_training_preprocessing/`  
  Dataset exploration, preprocessing, and prediction postprocessing notebooks.
- `02_finding_the_longest_only_growing_subsequence/`  
  Analysis for extracting the longest lesion growth subsequences.
- `03_new_patient_fit/`  
  Trying to fit the embedding for a lesion trajectory, that was not in the training trajectory.  
- `04_train_on_single_trajectory/`  
  Single-trajectory training to see if the model can fit a single patient perfectly.
- `05_data_visualizer/`  
  Visualize all lesion trajectories to find suitable ones for a evaluation dataset and the manual chosen ones for the evaluation dataset.
- `06_NaiveDilationBaseline/`  
  Baseline comparison experiments using a simple dilation-based model.
- `07_preprocessing_fixes/` 
  Follow-up experiments on the preprocessing and introducting of certain improvement
- `08_meta_learning/` 
  Try on getting meta learning to run, but it has not worked so far. 
- `09_plotting/`
  Creating the diagrams for the Report
- `10_loss_&_statistics/`  
  Collecting the statistics for the Report


### `configs/`

This folder stores all training configurations. It is the configuration layer that decides dataset parameters, training hyperparameters, loss settings, model class, and experiment name. These config files are python files as you can directly import ModelClasses or LossClasses via them:
- `00_default/`  
  Default training setup used by the standard INR pipeline. Contains also a config that serves as a starting point for creating new configs.
- `01_dialation/`  
  Configurations related to lesion dilation and patient-specific dilation variants.
- `02_tv_regulation/`  
  Configurations with INR + total-variation regularization. 
- `03_data_visualizer/`  
  Config for plotting all the lesion trajectories to manually check them and select a validation dataset.
- `04_lstm/`  
  LSTM-specific config.
- `05_meta_learning/`  
  Meta-learning training config.
- `06_neural_ode/`  
  Neural ODE config.
- `base_config.json`  
  Shared default settings loaded into configuration objects. This is useful when settings are added lateron that should apply for all configs for the train_*.py scripts to work, but you don't want to edit every single config file. 


### `utils/`

This is the core utility layer of the project. It contains most of the reusable data loaders, model definitions, dataset structures, losses, and plotting code. They are seperable into the following categories:

Dataset-Classes that are very useful for organizing all the different files from the dataset in classes like a MRI Scan, a Patient, a Lesion Trajectory or a Dataloader to directly access e.g. a lesion trajectories by providing a Patient ID and Lesion number:
- `paths.py`  
  Shared paths and project-wide file location helpers to keep the locations well-organized.
- `mri_dataloader.py`  
  Loader responsible for reading the MRI/trajectory data and caching lesion trajectories.
- `patient.py`  
  Patient metadata and helper logic associated with dataset items.
- `lesion_trajectory.py`  
  Defines trajectory-related logic for lesion across time.

Dataloader for Training:
- `dataloader.py`  
  Standard lesion dataset classes used by INR and related experiments.
- `dataloader_lstm.py`  
  Dataset and batching utilities for sequence-based LSTM training.

Models Implementations:
- `model_inr.py`  
  The main implicit neural representation model used in the default project pipeline.
- `model_lstm.py`  
  LSTM model implementation(s).
- `model_inr_meta.py`  
  Meta-learning INR model variants.
- `model_neural_ode.py`  
  Neural ODE model implementation.
- `model_dialation.py` and `model_dialation_patient_specific.py`  
  Models focused on lesion dilation or patient-specific adaptation.

Loss-functions:
- `loss_bce_dice.py`, `loss_bce_dice_tv.py`, `loss_bce_dice_tv_meta.py`, `loss_node.py`  
  Custom loss functions for segmentation and temporal prediction tasks, including BCE/Dice, TV regularization, and meta-learning variants.

Plotting gifs to see in-depth what the model is learning:
- `train_plotting.py` and `train_plotting_lstm.py`  
  Visualization utilities for plotting lesion evolution over time and evaluating model outputs.


### `data/`
This folder stores the project datasets and cached MRI-derived data.
- `entire_yale_dataset/`  
  Primary dataset directory containing the Yale imaging data used for lesion trajectory extraction and model training.
- `thomas_model_data/`  
  nnUnet model weights.


### `server/`

This folder contains the server-side setup for remote execution and containerized training. It includes:

- container definitions for reproducible environments
- server-side commands and instructions
- MLflow and artifact directories mirrored for remote runs
- notes for syncing, running jobs, and connecting to remote systems

### `slurm_jobs/`

This folder contains Slurm job submission scripts used for running training jobs on the computing server.

### `nnUNet/`

External nnUnet Repository

### `papers/`

This folder is intended for research papers, references, notes, or related literature. It is a place for publication material and background reading.


