#!/usr/bin/env python3
"""Submit a training job to SLURM for one config file.

The training script is chosen from the config's folder (one folder -> one script),
see TRAIN_SCRIPTS below. The job script is rendered from slurm_jobs/template.sbatch,
saved under slurm_jobs/submitted/, and handed to sbatch.

The Singularity image is looked up in SIF_CANDIDATES (home directory first,
then the project folder) or given with --sif.

Usage (from any folder, e.g. ~ or ~/projects/thesis):
    python projects/thesis/submit.py projects/thesis/configs/06_neural_ode/config.py
    python submit.py configs/09_point_lstm/config_sdf.py --dry-run
    python submit.py configs/06_neural_ode/config_large.py --sif ~/thesis.sif
"""
import argparse
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE = PROJECT_DIR / "slurm_jobs" / "template.sbatch"
SUBMITTED_DIR = PROJECT_DIR / "slurm_jobs" / "submitted"
# Where thesis.sif is looked for (first match wins) unless --sif is given.
SIF_CANDIDATES = [Path.home() / "thesis.sif", PROJECT_DIR / "thesis.sif"]

# Config folder (under configs/) -> training script (in the project root).
# Folders not listed here cannot be submitted.
TRAIN_SCRIPTS = {
    "01_dialation": "train_dilation.py",
    "02_tv_regulation": "train_inr.py",
    "06_neural_ode": "train_ode.py",
    "09_point_lstm": "train_point_lstm.py",
}

# Exceptions to the folder rule: "<folder>/<config file>" -> training script.
TRAIN_SCRIPT_OVERRIDES = {
    "06_neural_ode/config_patient_wise.py": "train_ode_patientwise.py",
}


def find_sif(explicit):
    if explicit:
        sif = Path(explicit).expanduser().resolve()
        if not sif.is_file():
            sys.exit(f"Singularity image not found: {sif}")
        return sif
    for candidate in SIF_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_job_script(config_rel: Path, train_script: str, job_name: str, sif: Path) -> str:
    template = TEMPLATE.read_text()
    replacements = {
        "JOB_NAME": job_name,
        "PROJECT_DIR": str(PROJECT_DIR),
        "SIF": str(sif),
        "TRAIN_SCRIPT": train_script,
        "CONFIG": config_rel.as_posix(),
    }
    for key, value in replacements.items():
        template = template.replace(f"@@{key}@@", value)
    if "@@" in template:
        sys.exit("Template still contains unfilled @@placeholders@@; check slurm_jobs/template.sbatch.")
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a training job for a config file.")
    parser.add_argument("config", help="config file, e.g. configs/06_neural_ode/config.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the job script but do not call sbatch")
    parser.add_argument("--sif", help="Singularity image (default: ~/thesis.sif, then <project>/thesis.sif)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        sys.exit(f"Config not found: {config_path}")
    try:
        config_rel = config_path.relative_to(PROJECT_DIR)
    except ValueError:
        sys.exit(f"Config must be inside the project folder {PROJECT_DIR}")

    if len(config_rel.parts) != 3 or config_rel.parts[0] != "configs":
        sys.exit("Config must be directly inside a folder under configs/, e.g. configs/06_neural_ode/config.py")
    folder = config_rel.parts[1]
    override_key = f"{folder}/{config_path.name}"

    if override_key in TRAIN_SCRIPT_OVERRIDES:
        train_script = TRAIN_SCRIPT_OVERRIDES[override_key]
    elif folder in TRAIN_SCRIPTS:
        train_script = TRAIN_SCRIPTS[folder]
    else:
        known = "\n  ".join(f"{f} -> {s}" for f, s in TRAIN_SCRIPTS.items())
        sys.exit(f"No training script assigned to configs/{folder}/.\nAssigned folders:\n  {known}\n"
                 "Add the folder to TRAIN_SCRIPTS in submit.py.")
    if not (PROJECT_DIR / train_script).is_file():
        sys.exit(f"Training script {train_script} does not exist.")

    sif = find_sif(args.sif)
    if sif is None:
        if not args.dry_run:
            sys.exit("thesis.sif not found in " + ", ".join(map(str, SIF_CANDIDATES)) + "; pass --sif.")
        sif = SIF_CANDIDATES[0]  # dry run on a machine without the image

    job_name = f"{folder}_{config_path.stem}"
    job_script = build_job_script(config_rel, train_script, job_name, sif)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    SUBMITTED_DIR.mkdir(parents=True, exist_ok=True)
    job_file = SUBMITTED_DIR / f"{job_name}_{timestamp}.sbatch"
    job_file.write_text(job_script)
    print(f"Job script: {job_file}")
    print(f"Training:   {train_script} --config {config_rel.as_posix()}")
    print(f"Image:      {sif}")
    print(f"Code/logs:  {PROJECT_DIR}")

    if args.dry_run:
        print("Dry run, not submitted.")
        return

    if shutil.which("sbatch") is None:
        sys.exit("sbatch not found. Run submit.py on the cluster, or use --dry-run.")

    # SLURM opens the output files before the job script runs, so the logs folder must exist now.
    (PROJECT_DIR / "logs").mkdir(exist_ok=True)
    subprocess.run(["sbatch", str(job_file)], check=True)


if __name__ == "__main__":
    main()
