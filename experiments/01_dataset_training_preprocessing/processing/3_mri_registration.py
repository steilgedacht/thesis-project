import concurrent.futures
import sys
import tqdm

sys.path.insert(1, '.')

from utils.mri_dataloader import MRI_Dataloader
from utils.patient import Patient

data_loader = MRI_Dataloader()

def register(patient_id):
    patient = Patient(patient_id)
    patient.register_all_to_first()
    return patient_id

if __name__ == "__main__":
    max_workers = 2

    # Switch to ProcessPoolExecutor to isolate ANTs/ITK C++ threads safely
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(register, patient): patient for patient in data_loader.patient_ids}
        for fut in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(data_loader.patient_ids)):
            try:
                print(fut.result())
            except Exception as e:
                patient_id = futures[fut]
                print(f"Error processing patient {patient_id}: {e}")