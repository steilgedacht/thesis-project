import tqdm as tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
sys.path.insert(1, '.')
import logging

from utils.patient import Patient
from utils.mri_dataloader import MRI_Dataloader

import gc

def process_and_save(input_path):
    print(f"Processing {input_path}")
    patient = Patient(input_path)
    if len(patient.patient_trajectory_paths) == 0 and len(patient.dates) > 2:
        patient.merge_lesion_to_trajectory()
    else:
        print(f"Skipped {input_path}")
    gc.collect()

SKIP_UNTIL = "YG_RCSHQXZDD8PB"

if __name__ == "__main__":
    data_loader = MRI_Dataloader()
    for patient in tqdm.tqdm(data_loader.patient_ids, total=len(data_loader.patient_ids)):
        if SKIP_UNTIL != "" and patient != SKIP_UNTIL:
            continue
        SKIP_UNTIL = "" 
        process_and_save(patient)
