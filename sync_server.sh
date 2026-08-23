#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="node"
REMOTE_PROJECT_DIR="~/projects/thesis"
LOCAL_SERVER_DIR="./server"
MLFLOW_PORT=5001
MERGE_SCRIPT="utils/merge_dbs.py"

# 1. Push code changes to the remote
echo "Syncing code to $REMOTE_HOST..."
rsync -avP --exclude='.git' --exclude='__pycache__/' --exclude='data' --exclude='.archive' \
      --exclude='mlruns' --exclude='mlflow.db' --exclude='mlartifacts' --exclude='nnunet_2' --exclude='server' \
      . "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"

# 2. Pull down only changed files
echo "Pulling MLflow data from $REMOTE_HOST..."
mkdir -p "$LOCAL_SERVER_DIR"
rsync -avP "${REMOTE_HOST}:~/projects/thesis/mlflow.db"    "${LOCAL_SERVER_DIR}/mlflow.db"
rsync -avP "${REMOTE_HOST}:~/projects/thesis/mlruns/"      "${LOCAL_SERVER_DIR}/mlruns/"
rsync -avP "${REMOTE_HOST}:~/projects/thesis/mlartifacts/" "${LOCAL_SERVER_DIR}/mlartifacts/"

# Get the absolute path to ensure MLflow handles SQLite paths accurately
ABS_LOCAL_DIR="$(cd "$LOCAL_SERVER_DIR" && pwd)"

# 3. Start a temporary local mlflow server to serve the copied data
echo "Starting temporary MLflow server on port $MLFLOW_PORT..."
mlflow server \
  --backend-store-uri "sqlite:///${ABS_LOCAL_DIR}/mlflow.db" \
  --default-artifact-root "${ABS_LOCAL_DIR}/mlartifacts" \
  --artifacts-destination "${ABS_LOCAL_DIR}/mlartifacts" \
  --host 127.0.0.1 --port "$MLFLOW_PORT" \
  --serve-artifacts &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping temporary MLflow server (PID $SERVER_PID)..."
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
# safety net only — catches errors/interrupts before we reach the explicit cleanup below
trap cleanup EXIT

# wait for the server to actually be up before hitting it
echo "Waiting for MLflow server to become ready..."
for i in {1..30}; do
  if curl -s "http://127.0.0.1:${MLFLOW_PORT}/health" >/dev/null 2>&1; then
    echo "Server is up."
    break
  fi
  sleep 1
done

# 4. Run the merge script
echo "Running merge script..."
python3 "$MERGE_SCRIPT"

# 5. Explicitly stop the temporary server now that the merge is done
cleanup
trap - EXIT   # disable the trap since cleanup already ran; avoids a harmless double-call

# 6. Ensure a persistent MLflow UI server is running in tmux
if tmux has-session -t mlflow_server 2>/dev/null; then
  echo "Persistent mlflow_server tmux session already running."
else
  echo "Starting persistent mlflow_server tmux session..."
  tmux new-session -d -s mlflow_server 'mlflow server'
fi

echo "Syncing complete."