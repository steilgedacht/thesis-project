"""
This package uses relative imports internally (`.paths`, `.patient`, ...)
so it works correctly when nested inside another project, e.g.:

    train.py
    utils/
        __init__.py   <- this file
        paths.py
        registrator.py
        data_sample.py
        lesion_trajectory.py
        mri_dataloader.py
        patient.py

`from utils.dataloader import LesionDataset` (or any other submodule)
triggers this __init__.py first, so every import here must be resolvable
without touching sys.path.

Import order still doesn't matter for the Patient <-> MRI_Dataloader <->
Lesion_Trajectory cycle: those three only reference each other inside
method bodies (lazy imports), never at module load time.
"""

from .paths import DatasetPaths, DEFAULT_DATA_PATH
from .registrator import Registrator
from .data_sample import DataSample
from .lesion_trajectory import Lesion_Trajectory
from .mri_dataloader import MRI_Dataloader
from .patient import Patient

__all__ = [
    "DatasetPaths",
    "DEFAULT_DATA_PATH",
    "Registrator",
    "DataSample",
    "Lesion_Trajectory",
    "MRI_Dataloader",
    "Patient",
]