rsync -avP --exclude='.git' --exclude='__pycache__/' --exclude='data' --exclude='mlruns' --exclude='mlflow.db' . node:~/projects/thesis
scp -r node:~/projects/thesis/mlflow.db ./server/mlflow.db
scp -r node:~/projects/thesis/mlruns ./server/mlruns
echo "Syncing complete." 