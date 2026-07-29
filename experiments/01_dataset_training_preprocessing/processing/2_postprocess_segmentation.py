import tqdm as tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.insert(1, '.')

from utils.mri_dataloader import MRI_Dataloader
from utils.patient import Patient


def process_and_save(input_path):
    patient = Patient(input_path)
    patient.process_samples()
    return patient.patient_id

if __name__ == "__main__":
    data_loader = MRI_Dataloader()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_and_save, patient): patient for patient in data_loader.patient_ids}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(data_loader.patient_ids)):
            try:
                result = future.result()
                print(f"Processed: {result}")
            except Exception as e:
                print(f"Error at {futures[future]}: {e}")