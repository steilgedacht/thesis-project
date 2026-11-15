#!/bin/bash
#SBATCH -n 8
#SBATCH -J validation_arena
#SBATCH --partition=full_optima
#SBATCH --mem=32G
#SBATCH --qos=normal_msc
#SBATCH --time=10:00:00
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
    bash -c "cd $WORKDIR && TMPDIR=$TMP_DIR python3.11 validation_arena.py --node-run-id fbebcf52c99c43b39228e42cf7539832 --lstm-run-id 8d67b8dd7050458fa6ea391e46296639 --inr-run-id e580908932ef4dcfa167c9a2bb300932 --output $TMP_DIR/validation_arena_results.csv --resolution 200" 