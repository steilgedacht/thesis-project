"""
MLflow metrics dashboard (Streamlit)

Install:  pip install streamlit mlflow pandas plotly
Run:      streamlit run mlflow_dashboard.py

Metric names are expected as  <method>_<eval>_<metric>, e.g.
    inr_interpolation_dice_score
    dilation_lesion_specific_extrapolation_dice_score
"""

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st

DEFAULT_URI = "http://localhost:5124"
DEFAULT_EXPERIMENT_ID = "9"  # Uniform Grid Validation
DEFAULT_EXPERIMENT_NAME = "Uniform Grid Validation"

# Order matters: longest suffixes first so "median_wrongly_..." is not
# mistaken for "wrongly_predicted_volume_cm3".
METRIC_NAMES = sorted(
    [
        "dice_score",
        "median_wrongly_predicted_volume_cm3",
        "average_wrongly_predicted_volume_cm3",
        "wrongly_predicted_volume_cm3",
    ],
    key=len,
    reverse=True,
)
EVAL_NAMES = ["embedding_trajectory", "interpolation", "extrapolation"]

st.set_page_config(page_title="MLflow metrics", layout="wide")
st.title("MLflow metrics explorer")


# ---------------------------------------------------------------- helpers
def parse_metric(name: str):
    """'inr_interpolation_dice_score' -> ('inr', 'interpolation', 'dice_score')"""
    for metric in METRIC_NAMES:
        if name.endswith("_" + metric):
            rest = name[: -len(metric) - 1]
            for ev in EVAL_NAMES:
                if rest.endswith("_" + ev):
                    return rest[: -len(ev) - 1], ev, metric
    return None


@st.cache_data(ttl=60, show_spinner="Loading experiments ...")
def load_experiments(uri: str):
    """{experiment_id: experiment_name}"""
    mlflow.set_tracking_uri(uri)
    return {e.experiment_id: e.name for e in mlflow.search_experiments()}


@st.cache_data(ttl=60, show_spinner="Loading runs ...")
def load_runs(uri: str, experiment_id: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(uri)
    return mlflow.search_runs(
        experiment_ids=[experiment_id],
        max_results=5000,
        order_by=["attributes.start_time DESC"],
    )


@st.cache_data(ttl=60, show_spinner="Loading run ...")
def load_runs_by_id(uri: str, run_ids: tuple) -> pd.DataFrame:
    """Load specific runs and mimic the layout of mlflow.search_runs()."""
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    rows = []
    for rid in run_ids:
        run = client.get_run(rid)
        row = {"run_id": run.info.run_id,
               "tags.mlflow.runName": run.data.tags.get("mlflow.runName")}
        row.update({f"metrics.{k}": v for k, v in run.data.metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def run_label(r: pd.Series) -> str:
    """Readable label for the run picker."""
    name = r.get("tags.mlflow.runName") or r["run_id"][:8]
    label = f"{name} ({r['run_id'][:6]})"
    start = r.get("start_time")
    if pd.notna(start):
        label += f" · {pd.Timestamp(start):%Y-%m-%d %H:%M}"
    status = r.get("status")
    if isinstance(status, str) and status != "FINISHED":
        label += f" · {status}"
    return label


def to_long(runs: pd.DataFrame) -> pd.DataFrame:
    """Wide MLflow dataframe -> long format (run, method, eval, metric, value)."""
    rows = []
    metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
    for _, r in runs.iterrows():
        run_name = r.get("tags.mlflow.runName") or r["run_id"][:8]
        label = f"{run_name} ({r['run_id'][:6]})"
        for col in metric_cols:
            val = r[col]
            if pd.isna(val):
                continue
            parsed = parse_metric(col.removeprefix("metrics."))
            if parsed:
                method, ev, metric = parsed
                rows.append(
                    dict(run=label, method=method, eval=ev, metric=metric, value=val)
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- sidebar: data source
with st.sidebar:
    st.header("Connection")
    uri = st.text_input("Tracking URI", DEFAULT_URI)
    if st.button("Refresh"):
        st.cache_data.clear()

    manual_ids_text = ""
    try:
        experiments = load_experiments(uri)
    except Exception as e:
        st.error(f"Could not load experiments from MLflow at {uri}\n\n{e}")
        st.stop()

    st.header("Experiment & Run")
    exp_ids = list(experiments)
    if DEFAULT_EXPERIMENT_ID in experiments:
        default_idx = exp_ids.index(DEFAULT_EXPERIMENT_ID)
    else:
        by_name = [i for i, n in experiments.items() if n == DEFAULT_EXPERIMENT_NAME]
        default_idx = exp_ids.index(by_name[0]) if by_name else 0
    exp_id = st.selectbox(
        "Experiment",
        exp_ids,
        index=default_idx,
        format_func=lambda i: f"{experiments[i]} (id {i})",
    )

    try:
        exp_runs = load_runs(uri, exp_id)
    except Exception as e:
        st.error(f"Could not load runs\n\n{e}")
        st.stop()

    if exp_runs.empty:
        st.warning("This experiment has no runs.")
        st.stop()

    labels = {r["run_id"]: run_label(r) for _, r in exp_runs.iterrows()}
    sel_run_ids = st.multiselect(
        "Run(s)",
        list(labels),
        default=list(labels)[:1],  # newest run
        format_func=lambda rid: labels[rid],
    )

    with st.expander("Or enter run ID(s) manually"):
        manual_ids_text = st.text_input("Run ID(s), comma-separated", "")

manual_ids = tuple(r.strip() for r in manual_ids_text.split(",") if r.strip())

try:
    if manual_ids:
        runs_raw = load_runs_by_id(uri, manual_ids)
    else:
        runs_raw = exp_runs[exp_runs["run_id"].isin(sel_run_ids)]
except Exception as e:
    st.error(f"Could not load runs from MLflow at {uri}\n\n{e}")
    st.stop()

if runs_raw.empty:
    st.info("Select at least one run in the sidebar.")
    st.stop()

df = to_long(runs_raw)
if df.empty:
    st.warning("No metrics matched the <method>_<eval>_<metric> naming scheme.")
    st.stop()

# ---------------------------------------------------------------- sidebar: filters
with st.sidebar:
    st.header("Filters")
    sel_methods = st.multiselect(
        "Methods", sorted(df["method"].unique()), default=sorted(df["method"].unique())
    )
    sel_evals = st.multiselect(
        "Evaluations", sorted(df["eval"].unique()), default=sorted(df["eval"].unique())
    )
    use_log = st.checkbox("Log y-axis", value=False)

df = df[df["method"].isin(sel_methods) & df["eval"].isin(sel_evals)]
if df.empty:
    st.info("Nothing selected.")
    st.stop()

multi = df["run"].nunique() > 1
EVAL_ORDER = {"eval": ["interpolation", "extrapolation", "embedding_trajectory"]}


def bar(metric: str, title: str, height: int = 380, y_max=None):
    d = df[df["metric"] == metric]
    fig = px.bar(
        d,
        x="method",
        y="value",
        color="eval",
        barmode="group",
        text_auto=".3g",
        facet_row="run" if multi else None,
        log_y=use_log,
        category_orders=EVAL_ORDER,
        title=title,
        height=height * (df["run"].nunique() if multi else 1),
    )
    if y_max is not None and not use_log:
        fig.update_yaxes(range=[0, y_max * 1.1])
    fig.update_layout(legend_title_text="")
    return fig


# ---------------------------------------------------------------- Dice
st.header("Dice")
st.caption("Hard Dice at threshold 0.5; higher is better")
st.plotly_chart(bar("dice_score", "Dice score"), width="stretch")

# ---------------------------------------------------------------- Volume
st.header("Wrongly predicted volume (cm³)")
st.caption("Lower is better. Median and average share the same y-scale.")
vol = df[df["metric"].isin(["median_wrongly_predicted_volume_cm3",
                            "average_wrongly_predicted_volume_cm3"])]
y_max = vol["value"].max() if not vol.empty else None

col_med, col_avg = st.columns(2)
col_med.plotly_chart(
    bar("median_wrongly_predicted_volume_cm3", "Median", y_max=y_max), width="stretch"
)
col_avg.plotly_chart(
    bar("average_wrongly_predicted_volume_cm3", "Average", y_max=y_max), width="stretch"
)
st.plotly_chart(
    bar("wrongly_predicted_volume_cm3", "Total wrongly predicted volume"), width="stretch"
)

# ---------------------------------------------------------------- Table
st.header("All values")
wide = df.pivot_table(index=["method", "eval", "run"], columns="metric", values="value")
st.dataframe(wide, width="stretch")
st.download_button("Download CSV", wide.to_csv().encode(), "metrics.csv", "text/csv")