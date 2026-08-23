import os
import glob
import sys
import tempfile
from multiprocessing import freeze_support
import torch
from tqdm import tqdm
import ants

os.chdir('/home/benjaminb/Dokumente/JKU/Semester_9/Practical_Work')
sys.path.insert(0, os.path.abspath('nnUNet'))

PATH_TO_DATASET = "data/entire_yale_dataset/PRE_POST_YBML"
OUTPUT_DATASET = "data/entire_yale_dataset/predictions"

def register_pre_to_post(pre_path: str, post_path: str, output_path: str) -> str:
    """Rigidly registers PRE (moving) to POST (fixed) using ANTs and saves the warped image."""
    fixed_image = ants.image_read(post_path)
    moving_image = ants.image_read(pre_path)

    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='Rigid'
    )

    ants.image_write(registration['warpedmovout'], output_path)
    return output_path

def main():
    os.environ["nnUNet_preprocessed"] = ".archive/train_nnUNet/train_dataset/data/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "data/thomas_model_data/nnunet/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
    os.environ["nnUNet_raw"] = ".archive/train_nnUNet/train_dataset/data/Dataset001_UCSFBrainMet"

    from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # 1. Discover all valid PRE/POST pairs
    post_files = sorted(glob.glob(f'{PATH_TO_DATASET}/**/*POST.nii.gz', recursive=True))
    
    cases_to_process = []

    print("Scanning dataset and checking existing predictions...")
    for post_path in post_files:
        pre_path = post_path.replace('POST.nii.gz', 'PRE.nii.gz')
        
        if not os.path.exists(pre_path):
            print(f"Skipping {post_path}: PRE contrast scan not found.")
            continue

        # Preserve subfolder structure relative to dataset root
        rel_path = os.path.relpath(os.path.dirname(post_path), PATH_TO_DATASET)
        save_dir = os.path.join(OUTPUT_DATASET, rel_path)
        os.makedirs(save_dir, exist_ok=True)
        
        output_file_prefix = os.path.join(save_dir, "seg_nnUnet")
        
        # Skip if output .nii.gz already exists
        if os.path.exists(f"{output_file_prefix}.nii.gz"):
            continue

        cases_to_process.append((pre_path, post_path, output_file_prefix))

    print(f"Found {len(cases_to_process)} cases ready for processing.")

    if not cases_to_process:
        print("No cases to process. All outputs already exist!")
        return

    # 2. Initialize Predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.2,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device('cuda'),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    predictor.initialize_from_trained_model_folder(
        os.environ["nnUNet_results"],
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name='checkpoint_final.pth',
    )
    predictor.allowed_mirroring_axes = (0, 1, 2)

    # 3. Sequential Prediction Loop with ANTs Registration
    print("Starting sequential registration and inference...")
    for pre_path, post_path, output_prefix in tqdm(cases_to_process, desc="Processing", unit="scan"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Register PRE to POST space
                reg_pre_path = os.path.join(tmp_dir, "PRE_registered.nii.gz")
                register_pre_to_post(pre_path, post_path, reg_pre_path)

                # Channel 0: Registered PRE, Channel 1: POST
                predictor.predict_from_files(
                    [[reg_pre_path, post_path]],
                    [output_prefix],
                    save_probabilities=False,
                    overwrite=False,
                    num_processes_preprocessing=1,       # 0 = Run in main process (low RAM)
                    num_processes_segmentation_export=1, # 0 = Run in main process (low RAM)
                    folder_with_segs_from_prev_stage=None,
                    num_parts=1,
                    part_id=0
                )
            except Exception as e:
                print(f"\n[ERROR] Failed processing {post_path}: {e}")
                continue

if __name__ == '__main__':
    freeze_support()
    main()