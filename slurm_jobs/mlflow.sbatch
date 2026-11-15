#!/bin/bash
#SBATCH -n 2
#SBATCH -J mlflow
#SBATCH --partition=full_optima
#SBATCH --mem=8G
#SBATCH --exclude=cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2
#SBATCH --qos=longrunning_msc
#SBATCH --time=169:00:00
#SBATCH --output=/home/optima/bbergmann/projects/thesis/logs/inr_%j.out
#SBATCH --error=/home/optima/bbergmann/projects/thesis/logs/inr_%j.err

set -e

WORKDIR=/home/optima/bbergmann
SIF=$WORKDIR/thesis.sif

# Relative paths (./artifacts, mlflow store) resolve from here, so your existing data is reused
cd "$WORKDIR"

echo "MLflow running on $(hostname):5124"
hostname > "$WORKDIR/mlflow_host.txt"   # so training jobs can find the server

singularity exec \
    --bind "$WORKDIR":"$WORKDIR" \
    --pwd "$WORKDIR" \
    "$SIF" \
    mlflow server \
        --host 0.0.0.0 \
        --port 5124 \
        --default-artifact-root ./artifacts \
        --allowed-hosts "*" \
        --disable-security-middleware