import tqdm as tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.insert(1, '.')

from utils.patient import Patient
from utils.mri_dataloader import MRI_Dataloader

import gc

def process_and_save(input_path):
    patient = Patient(input_path)
    patient.merge_lesion_to_trajectory()
    gc.collect()

if __name__ == "__main__":
    data_loader = MRI_Dataloader()

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(process_and_save, patient): patient for patient in data_loader.patient_ids}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(data_loader.patient_ids)):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error at {futures[future]}: {e}")