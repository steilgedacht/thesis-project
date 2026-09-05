import os

# Set thread limit FIRST before loading ITK/ANTs
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

import glob
import sys
import shutil
import tempfile
import queue
from threading import Thread
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support, cpu_count
import torch
from tqdm import tqdm
import ants

sys.path.insert(0, os.path.abspath('nnUNet'))

PATH_TO_DATASET = "data/entire_yale_dataset/PRE_POST_YBML"
OUTPUT_DATASET = "data/entire_yale_dataset/predictions"

def register_single_case(args):
    """Worker task executed in CPU ProcessPoolExecutor."""
    pre_path, post_path, tmp_pre_path, output_prefix = args
    try:
        if not os.path.exists(tmp_pre_path):
            fixed_image = ants.image_read(post_path)
            moving_image = ants.image_read(pre_path)

            registration = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform='Rigid'
            )
            ants.image_write(registration['warpedmovout'], tmp_pre_path)
        return tmp_pre_path, post_path, output_prefix, True
    except Exception as e:
        print(f"\n[ERROR] Registration failed for {pre_path}: {e}")
        return tmp_pre_path, post_path, output_prefix, False

def gpu_consumer_worker(job_queue, total_cases, results_dir):
    """Consumer thread running GPU inference sequentially on completed registrations."""
    from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    print("\n[GPU] Initializing nnU-Net Predictor on CUDA...")
    predictor = nnUNetPredictor(
        tile_step_size=0.2,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
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

    pbar = tqdm(total=total_cases, desc="Overall Progress (Registered & Predicted)", unit="scan")

    while True:
        item = job_queue.get()
        if item is None:  # Sentinel to shut down consumer
            job_queue.task_done()
            break

        tmp_pre_path, post_path, output_prefix = item

        try:
            # Predict and immediately write .nii.gz to output folder
            predictor.predict_from_files(
                [[tmp_pre_path, post_path]],
                [output_prefix],
                save_probabilities=False,
                overwrite=False,
                num_processes_preprocessing=4,
                num_processes_segmentation_export=4,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )
        except Exception as e:
            print(f"\n[ERROR] Inference failed for {output_prefix}: {e}")
        finally:
            # Clean up the single intermediate NIfTI file right after prediction
            if os.path.exists(tmp_pre_path):
                try:
                    os.remove(tmp_pre_path)
                except OSError:
                    pass
            pbar.update(1)
            job_queue.task_done()

    pbar.close()

def main():
    os.environ["nnUNet_preprocessed"] = ".archive/train_nnUNet/train_dataset/data/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "data/thomas_model_data/BratsMets/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Dataset001_UCSFBrainMet/nnUNetTrainer__nnUNetPlans__3d_fullres"
    os.environ["nnUNet_raw"] = ".archive/train_nnUNet/train_dataset/data/Dataset001_UCSFBrainMet"

    # 1. Discover unprocessed cases
    post_files = sorted(glob.glob(f'{PATH_TO_DATASET}/**/*POST.nii.gz', recursive=True))
    unprocessed_cases = []

    print("Scanning dataset and skipping finished predictions...")
    for post_path in post_files:
        pre_path = post_path.replace('POST.nii.gz', 'PRE.nii.gz')
        if not os.path.exists(pre_path):
            continue

        rel_path = os.path.relpath(os.path.dirname(post_path), PATH_TO_DATASET)
        save_dir = os.path.join(OUTPUT_DATASET, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        output_prefix = os.path.join(save_dir, "seg_nnUnet_2")
        if os.path.exists(f"{output_prefix}.nii.gz"):
            continue

        unprocessed_cases.append((pre_path, post_path, output_prefix))

    total_cases = len(unprocessed_cases)
    print(f"Found {total_cases} cases left to process.")
    if total_cases == 0:
        return

    # Scratch directory setup
    base_tmp_dir = os.environ.get("SLURM_TMPDIR", os.environ.get("TMPDIR", "./tmp_registration"))
    tmp_dir = tempfile.mkdtemp(prefix="queue_reg_", dir=base_tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # 2. Queue and Consumer Setup
    job_queue = queue.Queue(maxsize=30)  # Bound queue size to save RAM/disk space
    
    # Start GPU Inference Consumer Thread
    gpu_thread = Thread(target=gpu_consumer_worker, args=(job_queue, total_cases, tmp_dir), daemon=True)
    gpu_thread.start()

    # 3. CPU Parallel Registration Producer
    num_cpu_workers = min(16, max(1, cpu_count() // 2))
    print(f"Starting registration pipeline with {num_cpu_workers} CPU workers...")

    try:
        with ProcessPoolExecutor(max_workers=num_cpu_workers) as executor:
            futures = []
            for idx, (pre_path, post_path, output_prefix) in enumerate(unprocessed_cases):
                tmp_pre_path = os.path.join(tmp_dir, f"reg_{idx:05d}.nii.gz")
                task_args = (pre_path, post_path, tmp_pre_path, output_prefix)
                futures.append(executor.submit(register_single_case, task_args))

            # As registrations complete, push them to the GPU queue immediately
            for future in as_completed(futures):
                tmp_pre_path, post_path, output_prefix, success = future.result()
                if success:
                    job_queue.put((tmp_pre_path, post_path, output_prefix))

        # Signal consumer thread that production is done
        job_queue.put(None)
        gpu_thread.join()

    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == '__main__':
    freeze_support()
    main()