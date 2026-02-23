from mri_dataloader import *
import os

dataloader = MRI_Dataloader()

import concurrent.futures

def _process_patient(patient):
    if all(sample.registered_transform is not None for sample in patient.samples):
        return f"Skipping patient {patient.patient_id} (already registered)"
    try:
        print(f"Processing patient {patient.patient_id}...")
        patient.register_all_to_first()
        return f"Finished patient {patient.patient_id}"
    except Exception as e:
        return f"Error processing patient {patient.patient_id}: {e}"

patients = list(dataloader.iterate_patients())
max_workers = min(8, (os.cpu_count() or 1))

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_process_patient, p): p for p in patients}
    for fut in concurrent.futures.as_completed(futures):
        print(fut.result())
