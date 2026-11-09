#!/bin/bash
#SBATCH -n 8
#SBATCH -J inr_train
#SBATCH --partition=full_optima
#SBATCH --mem=32G
#SBATCH --qos=normal_msc
#SBATCH --time=1:00:00
#SBATCH --output=/home/optima/bbergmann/projects/thesis/logs/inr_%j.out
#SBATCH --error=/home/optima/bbergmann/projects/thesis/logs/inr_%j.err

set -e

# Make sure the log directory exists
mkdir -p /home/optima/bbergmann/projects/thesis/logs

# Start the instance fresh for this job (unique name per job to avoid clashes
# if you submit multiple jobs concurrently)
INSTANCE_NAME="thesis_env_${SLURM_JOB_ID}"

singularity instance start --nv thesis.sif "$INSTANCE_NAME"

# Make sure the instance gets cleaned up when the job ends, even on failure
trap 'singularity instance stop "$INSTANCE_NAME"' EXIT

singularity exec --nv instance://"$INSTANCE_NAME" \
    bash -c "cd /home/optima/bbergmann/projects/thesis && python3.11 new_patient_fit.py"