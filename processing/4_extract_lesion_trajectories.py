import glob
import tqdm as tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from mri_dataloader import Patient, MRI_Dataloader

def process_and_save(input_path):
    patient = Patient(input_path)
    patient.merge_lesion_to_trajectory()

if __name__ == "__main__":
    data_loader = MRI_Dataloader()

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_and_save, patient): patient for patient in data_loader.patient_ids}
        
        for future in tqdm.tqdm(as_completed(futures), total=len(data_loader.patient_ids)):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error at {futures[future]}: {e}")