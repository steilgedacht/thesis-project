#!/bin/bash
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -J validation_arena
#SBATCH --partition=full_optima
#SBATCH --mem=64G
#SBATCH --exclude=cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2
#SBATCH --qos=normal_msc
#SBATCH --time=5:00:00
#SBATCH --output=/home/optima/bbergmann/projects/thesis/logs/inr_%j.out
#SBATCH --error=/home/optima/bbergmann/projects/thesis/logs/inr_%j.err

set -e

# Make sure the log directory exists
mkdir -p /home/optima/bbergmann/projects/thesis/logs
WORKDIR="/home/optima/bbergmann/projects/thesis"
TMP_DIR="${WORKDIR}/tmp_validation_${SLURM_JOB_ID}"
mkdir -p "$TMP_DIR"
INSTANCE_STARTED=0

# Start the instance fresh for this job (unique name per job to avoid clashes
# if you submit multiple jobs concurrently)
INSTANCE_NAME="thesis_env_${SLURM_JOB_ID}"

cleanup() {
    status=$?
    trap - EXIT
    if [ "$INSTANCE_STARTED" -eq 1 ]; then
        singularity instance stop "$INSTANCE_NAME" || \
            echo "Warning: failed to stop instance $INSTANCE_NAME" >&2
    fi
    rm -rf -- "$TMP_DIR"
    exit "$status"
}
trap cleanup EXIT

singularity instance start --nv thesis.sif "$INSTANCE_NAME"
INSTANCE_STARTED=1

singularity exec --nv instance://"$INSTANCE_NAME" \
    bash -c 'cd "$WORKDIR" && TMPDIR="$TMP_DIR" python3.11 validation_arena.py \
        --node-run-id f82c0dc523134b0699c17c7b16e7a3d7 \
        --lstm-run-id 8d67b8dd7050458fa6ea391e46296639 \
        --inr-run-id 556437cb22ec45b09a60a74befe7cffa \
        --output "$TMP_DIR/validation_arena_results.csv" \
        --resolution 200 \
        --plot-contact-sheets'