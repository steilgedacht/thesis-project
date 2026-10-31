#!/bin/bash
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -J rnn_train
#SBATCH --partition=full_optima
#SBATCH --mem=64G
#SBATCH --exclude=cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2
#SBATCH --qos=longrunning_msc
#SBATCH --time=169:00:00
#SBATCH --output=/home/optima/bbergmann/projects/thesis/logs/rnn_%j.out
#SBATCH --error=/home/optima/bbergmann/projects/thesis/logs/rnn_%j.err

set -e

mkdir -p /home/optima/bbergmann/projects/thesis/logs

INSTANCE_NAME="thesis_env_${SLURM_JOB_ID}"
singularity instance start --nv thesis.sif "$INSTANCE_NAME"
trap 'singularity instance stop "$INSTANCE_NAME"' EXIT

singularity exec --nv instance://"$INSTANCE_NAME" \
    bash -c "cd /home/optima/bbergmann/projects/thesis && python3.11 train_rnn.py --config configs/08_rnn/config.py"
