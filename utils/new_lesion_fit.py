"""Shared protocol for fitting a trained model to a NEW lesion (embedding only).

Used by new_patient_fit.py and validation_arena.py so both run the same test:

    * Lesion: NEW_LESION (YG_69FMUN9PHHKE_2). MRI_Dataloader skips it when
      caching training lesions and its patient has no other trained lesion.
    * Split: the last ``heldout_scans`` scans are held out; the embedding is
      fitted on the earlier (support) scans only and scored on the held-out ones.
    * Basis: the convex-combination phase mixes only embedding rows that were
      trained (the NODE table has ~133k rows, 238 of them trained), never the
      lesion's own row, starting from their mean.
    * Region prior: the box comes from the support scans only.
    * Mode: fitted in train mode (as the model was trained, e.g. no region prior
      if region_prior_in_training=False), evaluated in eval mode.
"""

import json

import torch

from .mri_dataloader import MRI_Dataloader
from .paths import DatasetPaths
from .region_prior import mask_box

NEW_LESION = ("YG_69FMUN9PHHKE", 2)
HELDOUT_SCANS = 2
PROTOCOL = "new-lesion-v1: support/held-out split, trained-row basis, support box"


def lesion_dates(trajectory):
    return [str(d) for d in (
        trajectory.allowed_dates if hasattr(trajectory, "allowed_dates") else trajectory.dates
    )]


def split_support_heldout(dates, heldout_scans=HELDOUT_SCANS):
    """(support dates, held-out dates); at least two support scans."""
    if not 1 <= heldout_scans <= len(dates) - 2:
        raise ValueError(
            f"{len(dates)} scans cannot be split into >= 2 support and "
            f"{heldout_scans} held-out scan(s)."
        )
    return list(dates[:-heldout_scans]), list(dates[-heldout_scans:])


def _patient_to_idx():
    with open(DatasetPaths().patient_to_idx(), "r") as f:
        return json.load(f)


def global_lesion_id(trajectory, patient_to_idx=None):
    """Embedding id used by the datasets: patient_idx * 100 + label_id."""
    patient_to_idx = patient_to_idx or _patient_to_idx()
    return int(patient_to_idx[trajectory.patient_id]) * 100 + int(trajectory.label_id)


_TRAINING_IDS = {}


def training_lesion_ids(n_scans=6, only_growing=True):
    """Global ids of the lesions a model trained with these settings has seen."""
    key = (int(n_scans), bool(only_growing))
    if key not in _TRAINING_IDS:
        dataloader = MRI_Dataloader()
        dataloader.cache_lesion_trajectories_from_n_scans(n_scans=key[0], only_growing=key[1])
        patient_to_idx = _patient_to_idx()
        _TRAINING_IDS[key] = sorted(
            {global_lesion_id(t, patient_to_idx) for t in dataloader.cache_lesion_trajectories}
        )
    return list(_TRAINING_IDS[key])


def basis_rows(model, embedding_module, training_ids, exclude_id=None):
    """Rows of ``embedding_module`` that hold trained lesion embeddings.

    INR: ids are mapped through ``model.embedding_rows`` (compact table).
    NODE: the table is indexed by the global id directly.
    Other layouts (table smaller than the id space): all rows.
    """
    ids = [i for i in training_ids if i != exclude_id]
    if not ids:
        raise ValueError("No trained embedding rows left for the basis.")
    rows_buffer = getattr(model, "embedding_rows", None)
    if rows_buffer is not None:
        rows = torch.unique(rows_buffer[torch.tensor(ids, device=rows_buffer.device)])
        if exclude_id is not None and exclude_id < len(rows_buffer) and exclude_id in training_ids:
            rows = rows[rows != rows_buffer[exclude_id]]
        return rows.cpu()
    if max(ids) < embedding_module.num_embeddings:
        return torch.tensor(ids, dtype=torch.long)
    return torch.arange(embedding_module.num_embeddings)


def uniform_coefficients(count, device):
    """Start of the convex-combination phase: the mean of the basis."""
    return torch.full((count,), 1.0 / count, device=device)


def support_box(trajectory, support_dates):
    """Region-prior box around the support scans only."""
    return mask_box(trajectory, support_dates)
