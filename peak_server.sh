#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="node"
REMOTE_PROJECT_DIR="~/projects/thesis"
LOCAL_SERVER_DIR="./server"
MLFLOW_PORT=5001
MERGE_SCRIPT="utils/merge_dbs.py"

# 2. Pull down only changed files
echo "Pulling MLflow data from $REMOTE_HOST..."
mkdir -p "$LOCAL_SERVER_DIR"
rsync -avP "${REMOTE_HOST}:~/mlflow.db"    "${LOCAL_SERVER_DIR}/mlflow.db"
rsync -avP "${REMOTE_HOST}:~/mlartifacts/" "${LOCAL_SERVER_DIR}/mlartifacts/"

cd "$LOCAL_SERVER_DIR"
mlflow server \
  --backend-store-uri "sqlite:///mlflow.db" \
  --default-artifact-root "mlartifacts" \
  --artifacts-destination "mlartifacts" \
  --host "0.0.0.0" \
  --port "$MLFLOW_PORT"