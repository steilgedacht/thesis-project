## Connect to server

```bash
vpn-med
ssh node
```

## Start the instance at the server

Sync the local folder up
```bash
rsync -avP --exclude='.git' --exclude='__pycache__/' --exclude='venv/' --exclude='data' --exclude='server/mlartifacts' --exclude='mlruns' --exclude='mlflow.db' --exclude='.archive' . node:~/projects/thesis
```

Get some memory
```bash
salloc -n8 --gres=gpu:1 -J singularity_bash --partition=full_optima --mem=64G --exclude="cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2" --qos normal_msc --time=48:00:00 srun --pty /bin/bash
```

Stop the existing singularity
```bash
singularity instance stop thesis_env
sudo singularity build thesis.sif container.def
```

Start the singularity
```bash
singularity instance start --nv thesis.sif thesis_env
```

Connect to the singularity
```bash
singularity shell --nv instance://thesis_env
singularity exec instance://thesis_env mlflow server --host 127.0.0.1 --port 5124 --disable-security-middleware
```

Start the python script
```bash
cd projects/thesis && python3.11 inr.py
python3.11 inr.py
```

SSH Port forwarding
```bash
ssh -L 5124:localhost:5124 node
ssh -L 5000:localhost:5000 node
```

```bash
mlflow server --host 0.0.0.0 --port 5124 --backend-store-uri ./mlruns --default-artifact-root ./artifacts --allowed-hosts "*" --disable-security-middleware
```

## Submitting a sbatch

```bash
cp projects/thesis/train.sbatch . && sbatch train.sbatch
TERM=xterm-256color watch squeue -u $USER
sh watch_logs.sh projects/thesis/logs projects/thesis/logs
```

## When the predictions and labels where renewed:

```sh
rsync -avP ./data/entire_yale_dataset/predictions/ node:~/projects/entire_yale_dataset/predictions
```

## Getting the data back from the server

```sh
vpn-med
ssh node
sh start_mlflow.sh
sh sync_server.sh
```

## Build Container for Segmentation

```sh
salloc -n8 --gres=gpu:1 -J singularity_bash --partition=full_optima --mem=64G --exclude="cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2" --qos normal_msc --time=48:00:00 srun --pty /bin/bash
cp projects/thesis/server/container_seg.def . && sudo singularity build thesis_seg.sif container_seg.def
```

Download the seg files
```sh
rsync -avm --include='*/' --include='seg_nnUnet.nii.gz' --exclude='*' node:~/projects/entire_yale_dataset/predictions ./data/entire_yale_dataset/predictions
```