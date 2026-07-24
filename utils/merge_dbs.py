import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType, Metric
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import tempfile, os

SOURCE_URI = "http://127.0.0.1:5001"
TARGET_URI = "sqlite:///mlflow.db"
TARGET_EXPERIMENT_NAME = "Server Runs"


BATCH_SIZE = 1000        # MLflow's log_batch hard limit for metrics per call
READ_WORKERS = 8         # threads for fetching metric history in parallel

src_client = MlflowClient(tracking_uri=SOURCE_URI)
tgt_client = MlflowClient(tracking_uri=TARGET_URI)

# 1. Create (or get) the target experiment
existing = tgt_client.get_experiment_by_name(TARGET_EXPERIMENT_NAME)
target_exp_id = existing.experiment_id if existing else tgt_client.create_experiment(TARGET_EXPERIMENT_NAME)

# 2. Build a set of already-migrated source run_ids so we can resume/skip
existing_target_runs = tgt_client.search_runs(
    experiment_ids=[target_exp_id],
    run_view_type=ViewType.ALL,
    max_results=50000,
)
already_migrated = {
    run.data.tags.get("mlflow.source_run_id")
    for run in existing_target_runs
    if run.data.tags.get("mlflow.source_run_id")
}
print(f"Found {len(already_migrated)} runs already migrated — will skip these.")


def fetch_history(run_id, key):
    return key, src_client.get_metric_history(run_id, key)


def log_metrics_batched(run_id, all_metrics):
    """all_metrics: list of mlflow.entities.Metric objects"""
    for i in range(0, len(all_metrics), BATCH_SIZE):
        chunk = all_metrics[i:i + BATCH_SIZE]
        tgt_client.log_batch(run_id, metrics=chunk)


src_experiments = src_client.search_experiments(view_type=ViewType.ALL)

skipped_count = 0
migrated_count = 0

for exp in tqdm(src_experiments, desc="Experiments", unit="exp"):
    runs = src_client.search_runs(
        experiment_ids=[exp.experiment_id],
        run_view_type=ViewType.ALL,
        max_results=50000,
    )

    for run in tqdm(runs, desc=f"  Runs in '{exp.name}'", unit="run", leave=False):
        old_run_id = run.info.run_id

        if old_run_id in already_migrated:
            skipped_count += 1
            continue  # already migrated, skip everything for this run

        new_run = tgt_client.create_run(
            experiment_id=target_exp_id,
            tags={**run.data.tags, "mlflow.source_run_id": old_run_id},
            run_name=run.info.run_name,
            start_time=run.info.start_time,
        )
        new_run_id = new_run.info.run_id

        # Params can go through log_batch too (max 100 per call), but usually few enough
        # that it's fine as-is. Batch them anyway for consistency/safety:
        param_items = list(run.data.params.items())
        for i in range(0, len(param_items), 100):
            chunk = param_items[i:i + 100]
            tgt_client.log_batch(
                new_run_id,
                params=[mlflow.entities.Param(k, v) for k, v in chunk],
            )

        # --- Parallel fetch of metric history ---
        metric_keys = list(run.data.metrics.keys())
        all_metric_entities = []

        with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
            futures = {pool.submit(fetch_history, old_run_id, key): key for key in metric_keys}
            for future in tqdm(as_completed(futures), total=len(futures),
                                desc="    Fetching metric history", unit="metric", leave=False):
                key, history = future.result()
                for m in history:
                    all_metric_entities.append(Metric(key=m.key, value=m.value, timestamp=m.timestamp, step=m.step))

        # --- Batched sequential write ---
        log_metrics_batched(new_run_id, all_metric_entities)

        # Tags
        tag_items = [(k, v) for k, v in run.data.tags.items() if not k.startswith("mlflow.")]
        for k, v in tag_items:
            tgt_client.set_tag(new_run_id, k, v)

        # Artifacts
        try:
            artifact_listing = src_client.list_artifacts(old_run_id)
        except Exception as e:
            print(f"  [warn] could not list artifacts for run {old_run_id}: {e}")
            artifact_listing = []

        if artifact_listing:
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    local_path = src_client.download_artifacts(old_run_id, ".", tmp_dir)
                    if os.path.isdir(local_path) and os.listdir(local_path):
                        tgt_client.log_artifacts(new_run_id, local_path)
                except Exception as e:
                    print(f"  [warn] failed to download/copy artifacts for run {old_run_id}: {e}")
        else:
            pass  # no artifacts for this run, nothing to copy

        tgt_client.set_terminated(
            new_run_id,
            status=run.info.status,
            end_time=run.info.end_time,
        )

        migrated_count += 1

print(f"Done. Migrated {migrated_count} new run(s), skipped {skipped_count} already-migrated run(s).")