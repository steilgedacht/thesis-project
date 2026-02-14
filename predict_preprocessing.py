import glob
import os
import nibabel as nib
import numpy as np
import tqdm as tqdm
from scipy import ndimage
from skimage.morphology import convex_hull_image
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_sample(path = "data/entire_yale_dataset/PRE_POST_YBML/YG_0Y74OO0HCJZA/2012-01-29/YG_0Y74OO0HCJZA_2012-01-29_10-24-39_POST.nii.gz"):
    SEG_PATH = os.path.join(*(path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["0.nii.gz"]))

    mri = nib.load(path).get_fdata()
    seg = nib.load(SEG_PATH).get_fdata()

    # we introduce bleeding, to connect nearby lesions
    seg_dilated = ndimage.binary_dilation(seg, iterations=5)
    # now we label connected lesions
    labeled_array, num_features = ndimage.label(seg_dilated)
    # we want to have the original lesion size again so we multiply with the original seg
    labeled_array = seg * labeled_array

    # now we want to fill out holes in each lesion, we do that with a convex hull
    for i in range(num_features):
        lesion = labeled_array == (i + 1)
        convex_hull = convex_hull_image(lesion)
        labeled_array[convex_hull] = i + 1

    # we calculate the sizes of each lesion
    sizes = ndimage.sum(seg, labeled_array, range(1, num_features + 1))

    # we save the position of the lesions in euclidean roation coordinates
    lesion_positions = ndimage.center_of_mass(seg, labeled_array, range(1, num_features + 1))

    # calculate global center of mass of the MRI to have a reference point that is similar also to the next scan of the same patient
    total_area = np.sum((mri > 0).astype(np.float32))

    return [num_features, sizes, sizes / total_area, lesion_positions, labeled_array]


def process_and_save(input_path):
    num_features, sizes, relative_lesion_sizes, lesion_positions, labeled_array = process_sample(input_path)
    save_path = os.path.join(*(input_path.replace("PRE_POST_YBML", "predictions").split(os.path.sep)[:-1] + ["label"]))
    np.savez(save_path, num_features=num_features, sizes=sizes, relative_lesion_sizes=relative_lesion_sizes, lesion_positions=lesion_positions, labeled_array=labeled_array)
    return input_path

if __name__ == "__main__":
    globs = glob.glob('data/entire_yale_dataset/PRE_POST_YBML/**/**/*POST.nii.gz', recursive=True)
    print(f"{len(globs)=}")
    globs = sorted(globs)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_and_save, path): path for path in globs}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(globs)):
            try:
                result = future.result()
                print(f"Processed: {result}")
            except Exception as e:
                print(f"Error at {futures[future]}: {e}")