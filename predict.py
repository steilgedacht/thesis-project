import torch
import os
from multiprocessing import freeze_support
import glob
import nibabel as nib
import shutil

PATH_TO_DATASET = "data/entire_yale_dataset/predictions/"

def main():
    nnUNet_preprocessed=".archive/train_nnUNet/train_dataset/data/nnUNet_preprocessed"
    nnUNet_results="data/thomas_data/BratsMets/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/Dataset001_UCSFBrainMet/nnUNetTrainer__nnUNetPlans__3d_fullres"
    nnUNet_raw=".archive/train_nnUNet/train_dataset/data/Dataset001_UCSFBrainMet"

    os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
    os.environ["nnUNet_results"] = nnUNet_results
    os.environ["nnUNet_raw"] = nnUNet_raw

    from batchgenerators.utilities.file_and_folder_operations import join
    from nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device('cuda'),
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results),
        use_folds=(0,1,2,3,4),
        checkpoint_name='checkpoint_final.pth',
    )

    predictor.allowed_mirroring_axes = (0,)

    globs = glob.glob('**/*POST.nii.gz', recursive=True)
    print(f"{len(globs)=}")
    globs = sorted(globs)

    for i, input_path in enumerate(globs):

        save_path = f"{PATH_TO_DATASET}{os.path.join(*os.path.dirname(input_path).split('/')[-2:])}/"
        if os.path.exists(save_path):  # in case the file was already predicted
            continue

        os.makedirs(save_path)
        os.makedirs(f"/tmp/files_for_prediction/{i}", exist_ok=True)
        
        output_file_1 = f"/tmp/files_for_prediction/{i}/" + "T1.nii.gz"
        output_file_2 = f"/tmp/files_for_prediction/{i}/" + "T1_gad.nii.gz"
        
        shutil.copyfile(input_path, output_file_1)
        shutil.copyfile(input_path, output_file_2)

        predictor.predict_from_files(
            [[output_file_1, output_file_2]],
            [save_path + "0"],
            save_probabilities=False, 
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

        shutil.rmtree(f"/tmp/files_for_prediction/{i}/")

if __name__ == '__main__':
    freeze_support()
    main()