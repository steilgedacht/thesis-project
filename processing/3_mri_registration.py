from mri_dataloader import *
import os
import tqdm as tqdm

dataloader = MRI_Dataloader()

import concurrent.futures

def _process_sample(sample):
    sample.zoom()
    return True

max_workers = min(8, (os.cpu_count() or 1))

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_process_sample, sample): sample for sample in dataloader}
    for fut in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(data_loader)):
        print(fut.result())
