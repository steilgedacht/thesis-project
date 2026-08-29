#!/bin/bash
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -J segmentation
#SBATCH --partition=full_optima
#SBATCH --mem=64G
#SBATCH --exclude=cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2
#SBATCH --qos=normal_msc
#SBATCH --time=48:00:00
#SBATCH --output=/home/optima/bbergmann/projects/thesis/logs/inr_%j.out
#SBATCH --error=/home/optima/bbergmann/projects/thesis/logs/inr_%j.err

set -e

# Make sure the log directory exists
mkdir -p /home/optima/bbergmann/projects/thesis/logs

# Start the instance fresh for this job (unique name per job to avoid clashes
# if you submit multiple jobs concurrently)
INSTANCE_NAME="thesis_env_${SLURM_JOB_ID}"

singularity instance start --nv thesis_seg.sif "$INSTANCE_NAME"

# Make sure the instance gets cleaned up when the job ends, even on failure
trap 'singularity instance stop "$INSTANCE_NAME"' EXIT

singularity exec --nv instance://"$INSTANCE_NAME" \
    bash -c "cd /home/optima/bbergmann/projects/thesis && python3.11 1_segment_mri_with_nnUnet_gpu_other_weights.py"