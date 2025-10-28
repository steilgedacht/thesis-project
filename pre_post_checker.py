import os

def is_in(files, ending):
    return any(ending in file for file in files)

def count_matching_pairs(root_dir):
    count = 0
    patient_count = {}
    for root, _, files in os.walk(root_dir):
        pre_file = 'PRE.nii.gz'
        post_file = 'POST.nii.gz'
        if is_in(files, pre_file) and is_in(files, post_file):
            count += 1
            print(f"Found matching pair in directory: {root}")
            patient_id = root.split(os.sep)[-2]
            if patient_id not in patient_count:
                patient_count[patient_id] = 0
            patient_count[patient_id] += 1
    return count, sorted(patient_count.values())

if __name__ == "__main__":
    root_directory = "data/Yale-Brain-Mets-Longitudinal/Yale-Brain-Mets-Longitudinal/"
    result, result2 = count_matching_pairs(root_directory)
    print(f"Total directories with matching pairs: {result}")
    print(f"Total directories with matching pairs: {len(result2)}")

    import matplotlib.pyplot as plt

    plt.plot(range(len(result2)), result2)
    plt.xlabel("Patient")
    plt.ylabel("Number of POST and PRE pairs")
    plt.savefig("patient_distribution.png")
