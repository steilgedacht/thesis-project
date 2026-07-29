## Files

```sh
data_exploration.ipynb  # contains first exploration
prediction_postprocessing.ipynb  # contains much more exporation, the rest was all then put more formally into /processing
├── processing
│   ├── 1_segment_mri_with_nnUnet.py  #  uses nnUnet to get the labels where a lesion is
│   ├── 2_postprocess_segmentation.py  # takes the labeled data and creates a convex hull onto them and other postprocessing techniques
│   ├── 3_mri_registration.py  # registers the brains by scaling them to 500x500x50 and cropping any background 
│   └── 4_extract_lesion_trajectories.py  # get for each patient each lesion trajectory and save it in a way so that it can be used for training. 
```

## How to use
```sh
python3 experiments/01_dataset_training_preprocessing/processing/1_segment_mri_with_nnUnet.py
python3 experiments/01_dataset_training_preprocessing/processing/2_postprocess_segmentation.py
python3 experiments/01_dataset_training_preprocessing/processing/3_mri_registration.py
python3 experiments/01_dataset_training_preprocessing/processing/4_extract_lesion_trajectories.py
```