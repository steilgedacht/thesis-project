Sync the local folder up
```bash
rsync -avP --exclude='.git' --exclude='__pycache__/' --exclude='venv/' --exclude='data' --exclude='mlruns' --exclude='mlflow.db' . node:~/projects/thesis
```

Get some memory
```bash
salloc -n8 --gres=gpu:1 -J singularity_bash --partition=full_optima --mem=64G --exclude="cn1,cn2,cn5,on1,vn1,cn6,on2,on3,vn2" --qos normal_msc --time=48:00:00 srun --pty /bin/bash
```

Stop the existing singularity
```bash
singularity instance stop my_dev_env
```

Start the singularity
```bash
singularity instance start --nv thesis.sif my_dev_env
```

Connect to the singularity
```bash
singularity shell --nv instance://my_dev_env
singularity exec instance://my_dev_env mlflow server --host 127.0.0.1 --port 5124 --disable-security-middleware
```

Start the python script
```bash
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