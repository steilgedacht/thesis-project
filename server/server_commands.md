Sync the local folder up
```bash
rsync -avP --exclude='.git' --exclude='__pycache__/' --exclude='venv/' --exclude='data' --exclude='mlruns'  . node:~/projects/thesis
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
```

Start the python script
```bash
python3.11 inr.py
```