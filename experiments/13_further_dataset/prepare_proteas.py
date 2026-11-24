"""Prepare the PROTEAS brain-metastasis dataset as lesion trajectories (Yale format).

Raw data (read only):   data/further_datasets/PROTEAS
Output:                 data/further_datasets/PROTEAS_processed

Steps (all in this file, each step is skipped if its output already exists):

1. Index: per case, the timepoints that have both BraTS images and an expert
   tumor mask. Dates come from the clinical sheet (baseline = "Brain Mets
   Imaging Date", fu1 = "6 weeks", fu2 = "3 months", fu3 = "6 months",
   fu4 = "9 months", fu5 = "12 months"). Dates that are not increasing are
   retried with day and month swapped; cases that still fail are skipped.
   The intervals since baseline are rounded to whole weeks and attached to an
   artificial start date (ARTIFICIAL_START), so every scan has a date string.
2. Registration: every follow-up T1C is registered rigidly (ANTs) to the
   baseline T1C of the case; the mask follows with nearest-neighbour
   interpolation (as experiments/01_dataset_training_preprocessing/processing/
   3_mri_registration.py does for Yale).
3. Lesions: metastasis = mask labels 1 (necrosis) + 2 (enhancing tumor), edema
   (label 3) excluded. Holes are filled per slice, connected components are
   lesions, tiny components are dropped.
4. Tracking: lesions are followed over time by voxel overlap with the Yale
   tracker (utils.patient.Patient._build_overlap_trajectories). Files are
   written in the Yale layout, so utils.lesion_trajectory.Lesion_Trajectory
   loads them unchanged:
       predictions/<patient>/<date>/trajectory.nii.gz        (lesion ids)
       predictions/<patient>/<date>/registered_proc_mri.nii.gz (registered T1C)
       predictions/<patient>/lesion_trajectories_<id>.npz
5. Duplicates: paired cases (e.g. P07a / P07b) can contain the same lesion
   on the same scans; such duplicates are dropped.
6. Growth phase: as for Yale training data, only the growth phase is kept
   (Lesion_Trajectory.extract_growth_phase). Lesions whose growth phase has at
   least MIN_GROWTH_SCANS scans are listed in embedding_lesions.json.
7. Plots and a summary (plots/, trajectories.csv, summary.json).

Usage (from the project root or anywhere):
    python experiments/13_further_dataset/prepare_proteas.py [--workers 4] [--redo]
"""
import argparse
import itertools
import json
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)  # Lesion_Trajectory builds paths from relative strings
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import binary_fill_holes

RAW = Path("data/further_datasets/PROTEAS")
OUT = Path("data/further_datasets/PROTEAS_processed")
PREDICTIONS = OUT / "predictions"
REGISTRATION = OUT / "registration"
PLOTS = OUT / "plots"

TIMEPOINT_COLUMNS = {
    "baseline": "Brain Mets Imaging Date",
    "fu1": "6 weeks",
    "fu2": "3 months",
    "fu3": "6 months",
    "fu4": "9 months",
    "fu5": "12 months",
}
ARTIFICIAL_START = date(2000, 1, 1)
METASTASIS_LABELS = (1, 2)      # 1 necrosis, 2 enhancing tumor (3 = edema, excluded)
MIN_LESION_VOXELS = 10          # 1 mm voxels -> 10 mm^3
MIN_TRACK_SCANS = 3             # trajectories written (Yale keeps > 2 scans)
MIN_GROWTH_SCANS = 4            # embedding test: >= 2 support + 2 held-out scans
DUPLICATE_DICE = 0.95


# ---------------------------------------------------------------------------
# 1. Index
# ---------------------------------------------------------------------------
def _case_root(case):
    nested = RAW / case / case
    return nested if (nested / "BraTS").is_dir() else RAW / case


def _cell_dates(value):
    """Candidate dates for one sheet cell: as read, and day/month swapped."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (pd.Timestamp, datetime)):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        text = str(value).strip()
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt).date()
                break
            except ValueError:
                continue
        else:
            return []
    candidates = [parsed]
    if parsed.day <= 12 and parsed.day != parsed.month:
        candidates.append(date(parsed.year, parsed.day, parsed.month))
    return candidates


def _resolve_dates(cells):
    """Pick one date per timepoint so that they strictly increase, preferring
    as few day/month swaps as possible. Returns (dates, n_swaps) or (None, None)."""
    options = [_cell_dates(c) for c in cells]
    if any(not o for o in options):
        return None, None
    best = None
    for choice in itertools.product(*[list(enumerate(o)) for o in options]):
        dates = [d for _, d in choice]
        if all(a < b for a, b in zip(dates, dates[1:])):
            swaps = sum(i for i, _ in choice)
            if best is None or swaps < best[1]:
                best = (dates, swaps)
    return best if best else (None, None)


def _dicom_dates(root):
    dicom = root / "DICOM"
    found = set()
    if dicom.is_dir():
        for d in dicom.iterdir():
            m = re.search(r"(\d{4}-\d{2}-\d{2})", d.name)
            if m:
                found.add(m.group(1))
    return found


def _sheet_row(sheet, case, root, timepoints):
    """Sheet row of a case. P31 / P33 exist only as P31a/P31b, P33a/P33b in the
    sheet: take the row whose dates match the case's DICOM scan dates."""
    if case in sheet.index:
        return case
    candidates = [c for c in sheet.index if isinstance(c, str) and re.fullmatch(case + r"[ab]", c)]
    dicom = _dicom_dates(root)
    best, best_hits = None, 0
    for c in candidates:
        dates, _ = _resolve_dates([sheet.loc[c, TIMEPOINT_COLUMNS[tp]] for tp in timepoints
                                   if _cell_dates(sheet.loc[c, TIMEPOINT_COLUMNS[tp]])])
        hits = len({d.isoformat() for d in dates} & dicom) if dates else 0
        if hits > best_hits:
            best, best_hits = c, hits
    return best


def build_index():
    sheet = pd.read_excel(RAW / "PROTEAS-Clinical_and_demographic_data.xlsx", header=1)
    sheet = sheet.set_index(sheet.columns[0])
    index, skipped = {}, {}
    for case in sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name.startswith("P")):
        root = _case_root(case)
        brats = {d.name.lower(): d for d in (root / "BraTS").iterdir() if d.is_dir()} if (root / "BraTS").is_dir() else {}
        mask_dir = next((d for d in root.iterdir() if d.is_dir() and d.name.lower().replace(" ", "_") == "tumor_segmentation"), None)
        masks = {}
        if mask_dir is not None:
            for f in mask_dir.glob("*.nii.gz"):
                m = re.search(r"_tumor_mask_(\w+)\.nii\.gz$", f.name, re.IGNORECASE)
                if m:
                    masks[m.group(1).lower()] = f
        timepoints = [tp for tp in TIMEPOINT_COLUMNS if tp in brats and tp in masks and (brats[tp] / "t1c.nii.gz").exists()]
        row = _sheet_row(sheet, case, root, timepoints)
        if row is None:
            skipped[case] = "not in clinical sheet (no row matches its DICOM dates)"
            continue
        cells = [sheet.loc[row, TIMEPOINT_COLUMNS[tp]] for tp in timepoints]
        have_date = [tp for tp, c in zip(timepoints, cells) if _cell_dates(c)]
        if len(have_date) < len(timepoints):
            timepoints, cells = have_date, [sheet.loc[row, TIMEPOINT_COLUMNS[tp]] for tp in have_date]
        if len(timepoints) < MIN_TRACK_SCANS:
            skipped[case] = f"only {len(timepoints)} timepoints with images, mask and date"
            continue
        dates, swaps = _resolve_dates(cells)
        if dates is None:
            skipped[case] = "dates not increasing, also not with day/month swapped"
            continue
        weeks = [int(round((d - dates[0]).days / 7.0)) for d in dates]
        if len(set(weeks)) < len(weeks):
            skipped[case] = "two scans in the same week"
            continue
        index[case] = {
            "root": str(root),
            "sheet_row": row,
            "timepoints": timepoints,
            "real_dates": [d.isoformat() for d in dates],
            "date_swaps": swaps,
            "weeks": weeks,
            "dates": [(ARTIFICIAL_START + timedelta(weeks=w)).isoformat() for w in weeks],
            "t1c": {tp: str(brats[tp] / "t1c.nii.gz") for tp in timepoints},
            "mask": {tp: str(masks[tp]) for tp in timepoints},
        }
    return index, skipped


# ---------------------------------------------------------------------------
# 2. Registration (one process per case)
# ---------------------------------------------------------------------------
def _brain_dice(a, b):
    a, b = a != 0, b != 0
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def register_case(case, entry, redo=False):
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
    import ants

    out_dir = REGISTRATION / case
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = entry["timepoints"][0]
    fixed = ants.image_read(entry["t1c"][baseline])
    fixed_np = fixed.numpy()
    quality = {}
    for tp in entry["timepoints"]:
        t1c_out, mask_out = out_dir / f"{tp}_t1c.nii.gz", out_dir / f"{tp}_mask.nii.gz"
        moving = ants.image_read(entry["t1c"][tp])
        mask = ants.image_read(entry["mask"][tp])
        before = _brain_dice(fixed_np, moving.numpy())
        if not redo and t1c_out.exists() and mask_out.exists():
            after = _brain_dice(fixed_np, ants.image_read(str(t1c_out)).numpy())
        elif tp == baseline:
            ants.image_write(moving, str(t1c_out))
            ants.image_write(mask, str(mask_out))
            after = before
        else:
            reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")
            warped_mask = ants.apply_transforms(
                fixed=fixed, moving=mask, transformlist=reg["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            ants.image_write(reg["warpedmovout"], str(t1c_out))
            ants.image_write(warped_mask, str(mask_out))
            after = _brain_dice(fixed_np, reg["warpedmovout"].numpy())
        quality[tp] = {"brain_dice_before": before, "brain_dice_after": after}
    return case, quality


# ---------------------------------------------------------------------------
# 3.-4. Lesions and tracking
# ---------------------------------------------------------------------------
def _metastasis_components(mask):
    met = np.isin(mask, METASTASIS_LABELS)
    for z in range(met.shape[2]):
        met[:, :, z] = binary_fill_holes(met[:, :, z])
    labeled, n = ndimage.label(met)
    if n:
        sizes = ndimage.sum_labels(met, labeled, range(1, n + 1))
        for label, size in enumerate(sizes, start=1):
            if size < MIN_LESION_VOXELS:
                labeled[labeled == label] = 0
        labeled, n = ndimage.label(labeled > 0)
    return labeled.astype(np.int32), n


def _scan_lesions(labeled, spacing):
    labels = np.unique(labeled)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return {"centroids": np.empty((0, 3)), "sizes": np.array([]), "labels": labels,
                "labeled_mask": labeled, "masks": [], "spacing": spacing}
    centroids = np.asarray(ndimage.center_of_mass(np.ones_like(labeled), labeled, labels)) * spacing
    sizes = np.asarray(ndimage.sum_labels(np.ones_like(labeled), labeled, labels), dtype=float)
    return {"centroids": centroids, "sizes": sizes, "labels": labels, "labeled_mask": labeled,
            "masks": [(labeled == label).astype(np.uint8) for label in labels], "spacing": spacing}


def track_case(case, entry):
    from utils.patient import Patient

    tracker = Patient.__new__(Patient)  # only the overlap tracker is used
    patient_id = f"PROTEAS_{case}"
    out_dir = PREDICTIONS / patient_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t1c_img = [nib.load(REGISTRATION / case / f"{tp}_t1c.nii.gz") for tp in entry["timepoints"]]
    affine = t1c_img[0].affine
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    brain_voxels = int((np.asarray(t1c_img[0].dataobj) != 0).sum())
    scan_lesions, lesion_counts = [], []
    for tp in entry["timepoints"]:
        mask = np.asarray(nib.load(REGISTRATION / case / f"{tp}_mask.nii.gz").dataobj).astype(np.int32)
        labeled, n = _metastasis_components(mask)
        scan_lesions.append(_scan_lesions(labeled, spacing))
        lesion_counts.append(n)

    shape = scan_lesions[0]["labeled_mask"].shape
    trajectories = tracker._build_overlap_trajectories(
        scan_lesions, len(scan_lesions), max_gap=7, motion_threshold=0.07 * max(shape)
    )
    trajectories = [t for t in trajectories if len(t["scan_indices"]) >= MIN_TRACK_SCANS]

    trajectory_maps = [np.zeros(shape, dtype=np.uint8) for _ in scan_lesions]
    written = []
    for traj_id, trajectory in enumerate(trajectories, start=1):
        voxels = []
        for scan_idx, (_, label) in zip(trajectory["scan_indices"], trajectory["labels"]):
            lesion = (scan_lesions[scan_idx]["labeled_mask"] == label) & (trajectory_maps[scan_idx] == 0)
            trajectory_maps[scan_idx][lesion] = traj_id
            voxels.append(int(lesion.sum()))
        dates = [entry["dates"][i] for i in trajectory["scan_indices"]]
        np.savez(
            out_dir / f"lesion_trajectories_{traj_id}.npz",
            patient_id=patient_id, label_id=traj_id, dates=dates, n_scans=len(dates),
            sizes=np.log(np.asarray(voxels) / (brain_voxels + 1e-7) + 1e-7),
        )
        written.append({"label_id": traj_id, "scan_indices": trajectory["scan_indices"], "voxels": voxels})

    for scan_idx, tp in enumerate(entry["timepoints"]):
        date_dir = out_dir / entry["dates"][scan_idx]
        date_dir.mkdir(exist_ok=True)
        nib.save(nib.Nifti1Image(trajectory_maps[scan_idx], affine), date_dir / "trajectory.nii.gz")
        nib.save(t1c_img[scan_idx], date_dir / "registered_proc_mri.nii.gz")
    return patient_id, lesion_counts, written


# ---------------------------------------------------------------------------
# 5. Duplicates between paired cases (P07a / P07b, ...)
# ---------------------------------------------------------------------------
def _trajectory_mask(patient_id, label_id, date_str):
    return np.asarray(nib.load(PREDICTIONS / patient_id / date_str / "trajectory.nii.gz").dataobj) == label_id


def find_duplicates(trajectories):
    """trajectories: list of dicts with patient_id, case_base, label_id, dates."""
    duplicates = set()
    by_base = {}
    for t in trajectories:
        by_base.setdefault(t["case_base"], []).append(t)
    for group in by_base.values():
        for a, b in itertools.combinations(group, 2):
            if a["patient_id"] == b["patient_id"] or (b["patient_id"], b["label_id"]) in duplicates:
                continue
            common = sorted(set(a["dates"]) & set(b["dates"]))
            if len(common) < MIN_TRACK_SCANS:
                continue
            # same real scans? (same artificial dates are not enough)
            if [a["real_by_date"][d] for d in common] != [b["real_by_date"][d] for d in common]:
                continue
            dices = []
            for d in common:
                ma, mb = _trajectory_mask(a["patient_id"], a["label_id"], d), _trajectory_mask(b["patient_id"], b["label_id"], d)
                dices.append(2 * (ma & mb).sum() / max(ma.sum() + mb.sum(), 1))
            if np.mean(dices) >= DUPLICATE_DICE:
                duplicates.add((b["patient_id"], b["label_id"]))
    return duplicates


# ---------------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------------
def _crop(center, shape, half=30):
    lo = [int(max(0, min(c - half, s - 2 * half))) for c, s in zip(center, shape)]
    return slice(lo[0], lo[0] + 2 * half), slice(lo[1], lo[1] + 2 * half)


def plot_case_overview(case, entry, quality, patient_id):
    n = len(entry["timepoints"])
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4), squeeze=False)
    maps = [np.asarray(nib.load(PREDICTIONS / patient_id / d / "trajectory.nii.gz").dataobj) for d in entry["dates"]]
    union = np.any(np.stack(maps) > 0, axis=0)
    z = int(np.argmax(union.sum(axis=(0, 1)))) if union.any() else maps[0].shape[2] // 2
    base = np.asarray(nib.load(REGISTRATION / case / f"{entry['timepoints'][0]}_t1c.nii.gz").dataobj)
    cmap = plt.get_cmap("tab10")
    for i, (tp, d) in enumerate(zip(entry["timepoints"], entry["dates"])):
        t1c = np.asarray(nib.load(PREDICTIONS / patient_id / d / "registered_proc_mri.nii.gz").dataobj)
        for row, image in enumerate((t1c, None)):
            ax = axes[row, i]
            if row == 0:
                ax.imshow(np.rot90(t1c[:, :, z]), cmap="gray")
                for label in np.unique(maps[i])[1:]:
                    ax.contour(np.rot90(maps[i][:, :, z] == label), levels=[0.5], colors=[cmap((label - 1) % 10)], linewidths=1)
                ax.set_title(f"{tp} · week {entry['weeks'][i]}\n{d}", fontsize=9)
            else:
                # registration check: baseline (magenta) vs this timepoint (green)
                a = base[:, :, z] / (np.percentile(base, 99) + 1e-6)
                b = t1c[:, :, z] / (np.percentile(t1c, 99) + 1e-6)
                rgb = np.clip(np.stack([a, b, a], axis=-1), 0, 1)
                ax.imshow(np.rot90(rgb))
                q = quality.get(tp, {})
                ax.set_title(f"vs baseline: brain Dice\n{q.get('brain_dice_before', np.nan):.3f} → {q.get('brain_dice_after', np.nan):.3f}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{patient_id}: metastasis lesions (colour = trajectory id), slice z={z}; "
                 "row 2: baseline magenta / timepoint green after registration", fontsize=10)
    fig.tight_layout()
    path = PLOTS / "cases" / f"{patient_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_lesion(row):
    pid, label = row["patient_id"], row["label_id"]
    dates, growth = row["dates"], set(row["growth_dates"])
    masks = [_trajectory_mask(pid, label, d) for d in dates]
    union = np.any(np.stack(masks), axis=0)
    center = np.array(ndimage.center_of_mass(union))
    z = int(round(center[2]))
    sl = _crop(center[:2], union.shape[:2])
    n = len(dates)
    fig = plt.figure(figsize=(2.6 * n, 5.4))
    grid = fig.add_gridspec(2, n, height_ratios=[1.1, 1])
    for i, d in enumerate(dates):
        ax = fig.add_subplot(grid[0, i])
        t1c = np.asarray(nib.load(PREDICTIONS / pid / d / "registered_proc_mri.nii.gz").dataobj)
        ax.imshow(np.rot90(t1c[sl[0], sl[1], z]), cmap="gray")
        if masks[i][sl[0], sl[1], z].any():
            ax.contour(np.rot90(masks[i][sl[0], sl[1], z]), levels=[0.5], colors=["#c62523"], linewidths=1.2)
        in_growth = d in growth
        ax.set_title(f"week {row['weeks'][i]}\n{row['volumes_cm3'][i]:.2f} cm³" + ("" if in_growth else "\n(not growth)"),
                     fontsize=8, color="black" if in_growth else "grey")
        for spine in ax.spines.values():
            spine.set_edgecolor("#34c759" if in_growth else "#bbbbbb")
            spine.set_linewidth(2.5 if in_growth else 1)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(grid[1, :])
    ax.plot(row["weeks"], row["volumes_cm3"], "o-", color="#bbbbbb", label="all scans")
    gw = [w for w, d in zip(row["weeks"], dates) if d in growth]
    gv = [v for v, d in zip(row["volumes_cm3"], dates) if d in growth]
    ax.plot(gw, gv, "o-", color="#34c759", label=f"growth phase ({len(gw)} scans)")
    ax.set_xlabel("weeks since baseline"); ax.set_ylabel("volume [cm³]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f"{pid} lesion {label}  (slice z={z}, 60 mm crop, green frame = growth phase)", fontsize=10)
    fig.tight_layout()
    path = PLOTS / "lesions" / f"{pid}_lesion_{label}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(table):
    usable = table[table["usable_for_embedding"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    bins = np.arange(0.5, max(7, table["n_scans"].max() + 1.5))
    axes[0].hist(table["n_scans"], bins=bins, alpha=0.5, label="all tracked lesions", color="#888888")
    axes[0].hist(table["n_growth_scans"], bins=bins, alpha=0.7, label="growth phase", color="#34c759")
    axes[0].axvline(MIN_GROWTH_SCANS - 0.5, color="#c62523", ls="--", label=f"embedding test needs ≥{MIN_GROWTH_SCANS}")
    axes[0].set_xlabel("scans per lesion trajectory"); axes[0].set_ylabel("lesions"); axes[0].legend(fontsize=8)
    axes[0].set_title("Trajectory lengths")
    counts = usable.groupby("patient_id").size().sort_values(ascending=False)
    axes[1].bar(range(len(counts)), counts.values, color="#34c759")
    axes[1].set_xticks(range(len(counts)), [c.replace("PROTEAS_", "") for c in counts.index], rotation=90, fontsize=8)
    axes[1].set_ylabel("usable lesions"); axes[1].set_title(f"Usable lesions per case ({len(usable)} total)")
    for _, r in usable.iterrows():
        gw = [w for w, d in zip(r["weeks"], r["dates"]) if d in set(r["growth_dates"])]
        gv = [v for v, d in zip(r["volumes_cm3"], r["dates"]) if d in set(r["growth_dates"])]
        axes[2].plot(gw, gv, "o-", ms=3, lw=1)
    axes[2].set_yscale("log"); axes[2].set_xlabel("weeks since baseline"); axes[2].set_ylabel("volume [cm³]")
    axes[2].set_title("Growth phases of the usable lesions"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "summary.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4, help="parallel registrations")
    parser.add_argument("--redo", action="store_true", help="redo registrations that already exist")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # 1. index
    index, skipped = build_index()
    print(f"[1] {len(index)} cases indexed, {len(skipped)} skipped")
    for case, reason in skipped.items():
        print(f"    skip {case}: {reason}")
    swapped = {c: e["date_swaps"] for c, e in index.items() if e["date_swaps"]}
    if swapped:
        print(f"    dates read with day/month swapped: {swapped}")
    (OUT / "index.json").write_text(json.dumps({"cases": index, "skipped": skipped}, indent=2))

    # 2. registration
    quality = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(register_case, c, e, args.redo): c for c, e in index.items()}
        for i, fut in enumerate(as_completed(futures), 1):
            case, q = fut.result()
            quality[case] = q
            worst = min(v["brain_dice_after"] for v in q.values())
            print(f"[2] registered {case} ({i}/{len(index)}), worst brain Dice to baseline {worst:.3f}")
    (OUT / "registration_quality.json").write_text(json.dumps(quality, indent=2))

    # 3.-4. lesions + tracking
    rows = []
    for case, entry in index.items():
        patient_id, lesion_counts, written = track_case(case, entry)
        print(f"[3] {patient_id}: lesions per scan {lesion_counts}, {len(written)} trajectories with ≥{MIN_TRACK_SCANS} scans")
        for w in written:
            idx = w["scan_indices"]
            rows.append({
                "patient_id": patient_id, "case": case, "case_base": re.sub(r"[ab]$", "", case),
                "label_id": w["label_id"],
                "dates": [entry["dates"][i] for i in idx],
                "weeks": [entry["weeks"][i] for i in idx],
                "timepoints": [entry["timepoints"][i] for i in idx],
                "real_by_date": {entry["dates"][i]: entry["real_dates"][i] for i in idx},
                "volumes_cm3": [v / 1000.0 for v in w["voxels"]],
                "n_scans": len(idx),
            })
        plot_case_overview(case, entry, quality[case], patient_id)

    # 5. duplicates
    duplicates = find_duplicates(rows)
    for pid, label in sorted(duplicates):
        (PREDICTIONS / pid / f"lesion_trajectories_{label}.npz").unlink()
    rows = [r for r in rows if (r["patient_id"], r["label_id"]) not in duplicates]
    print(f"[5] removed {len(duplicates)} duplicate trajectories: {sorted(duplicates)}")

    # 6. growth phase (same code path as for Yale training lesions)
    from utils.lesion_trajectory import Lesion_Trajectory
    for r in rows:
        trj = Lesion_Trajectory(load_from_trajectory_path=str(PREDICTIONS / r["patient_id"] / f"lesion_trajectories_{r['label_id']}.npz"))
        r["growth_dates"] = list(trj.allowed_dates)
        r["n_growth_scans"] = len(trj.allowed_dates)
        r["usable_for_embedding"] = r["n_growth_scans"] >= MIN_GROWTH_SCANS
        growth_idx = [r["dates"].index(d) for d in r["growth_dates"]]
        r["growth_weeks"] = [r["weeks"][i] for i in growth_idx]
        r["growth_volume_change"] = r["volumes_cm3"][growth_idx[-1]] / max(r["volumes_cm3"][growth_idx[0]], 1e-6)

    table = pd.DataFrame(rows).sort_values(["n_growth_scans", "n_scans"], ascending=False)
    table.drop(columns=["real_by_date"]).to_csv(OUT / "trajectories.csv", index=False)
    usable = table[table["usable_for_embedding"]]
    (OUT / "embedding_lesions.json").write_text(json.dumps([
        {"patient_id": r.patient_id, "label_id": int(r.label_id),
         "npz": str(PREDICTIONS / r.patient_id / f"lesion_trajectories_{r.label_id}.npz"),
         "growth_dates": r.growth_dates, "growth_weeks": r.growth_weeks}
        for r in usable.itertuples()
    ], indent=2))

    # 7. plots + summary
    for r in rows:
        if r["usable_for_embedding"] or r["n_scans"] >= MIN_GROWTH_SCANS:
            plot_lesion(r)
    plot_summary(table)
    summary = {
        "cases_indexed": len(index), "cases_skipped": skipped,
        "trajectories": int(len(table)), "duplicates_removed": len(duplicates),
        "scans_per_trajectory": table["n_scans"].value_counts().sort_index().to_dict(),
        "growth_scans_per_trajectory": table["n_growth_scans"].value_counts().sort_index().to_dict(),
        "usable_for_embedding": int(len(usable)),
        "usable_cases": int(usable["patient_id"].nunique()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int))
    print(json.dumps(summary, indent=2, default=int))
    print(f"\nOutputs in {OUT}: predictions/, plots/, trajectories.csv, embedding_lesions.json")


if __name__ == "__main__":
    main()
