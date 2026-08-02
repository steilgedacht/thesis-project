import os
import tqdm as tqdm
import sys
sys.path.insert(1, '.')

from utils.mri_dataloader import MRI_Dataloader
from utils.patient import Patient
import concurrent.futures


data_loader = MRI_Dataloader()


def register(patient_id):
    patient = Patient(patient_id)
    patient.register_all_to_first()
    return patient_id

max_workers = 4

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(register, patient): patient for patient in data_loader.patient_ids}
    for fut in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(data_loader)):
        print(fut.result())
