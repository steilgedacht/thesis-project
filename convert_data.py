import glob
import re
import matplotlib.pyplot as plt
from datetime import datetime
import os
import nibabel as nib
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import tqdm as tqdm
from scipy.ndimage import zoom

globs = glob.glob('**/*POST.nii.gz', recursive=True)
print(f"{len(globs)=}")


from concurrent.futures import ThreadPoolExecutor

def resample_to_target(input_path, output_path, target_shape=(240, 240, 155)):
    img = nib.load(input_path)
    data = img.get_fdata()
    
    current_shape = data.shape
    
    scale_factors = [t / c for t, c in zip(target_shape, current_shape)]    
    resampled_data = zoom(data, scale_factors, order=3)  # order=3 für kubische Interpolation
    
    # resampled_data = np.stack([resampled_data], axis=0)
    resampled_data = np.stack([resampled_data.transpose(2,0,1), resampled_data.transpose(2,0,1)], axis=0)
    new_img = nib.Nifti1Image(resampled_data, img.affine)
    nib.save(new_img, output_path)
    nib.save(new_img, output_path.replace('T1.nii.gz', 'T1_gad.nii.gz'))

def process_file(i, input_file):
    os.makedirs(f"/run/media/benjaminb/Volume/JKU/Semester_9/Practical_Work/files_for_prediction/{i}", exist_ok=True)
    output_file = f"/run/media/benjaminb/Volume/JKU/Semester_9/Practical_Work/files_for_prediction/{i}/" + f"T1.nii.gz"
    print(i)
    resample_to_target(input_file, output_file)

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [
        executor.submit(process_file, i, glob)
        for i, glob in enumerate(globs)
    ]
    for future in futures:
        future.result()
