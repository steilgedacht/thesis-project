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
DEFAULT_RUN_ID = "7ac3df95f6674199b297889000b27ce8"

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


@st.cache_data(ttl=60, show_spinner="Loading from MLflow ...")
def load_experiments(uri: str):
    mlflow.set_tracking_uri(uri)
    return {e.name: e.experiment_id for e in mlflow.search_experiments()}


@st.cache_data(ttl=60, show_spinner="Loading runs ...")
def load_runs(uri: str, experiment_id: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(uri)
    return mlflow.search_runs(experiment_ids=[experiment_id], max_results=5000)


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


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    uri = st.text_input("Tracking URI", DEFAULT_URI)
    if st.button("Refresh"):
        st.cache_data.clear()

with st.sidebar:
    run_ids_text = st.text_input(
        "Run ID(s), comma-separated (leave empty to browse an experiment)",
        DEFAULT_RUN_ID,
    )
run_ids = tuple(r.strip() for r in run_ids_text.split(",") if r.strip())

try:
    if run_ids:
        runs_raw = load_runs_by_id(uri, run_ids)
    else:
        experiments = load_experiments(uri)
        with st.sidebar:
            exp_name = st.selectbox("Experiment", list(experiments))
        runs_raw = load_runs(uri, experiments[exp_name])
except Exception as e:
    st.error(f"Could not load from MLflow at {uri}\n\n{e}")
    st.stop()

if runs_raw.empty:
    st.warning("No runs found.")
    st.stop()

df = to_long(runs_raw)
if df.empty:
    st.warning("No metrics matched the <method>_<eval>_<metric> naming scheme.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    all_runs = sorted(df["run"].unique())
    sel_runs = st.multiselect("Runs", all_runs, default=all_runs[:5])
    sel_methods = st.multiselect(
        "Methods", sorted(df["method"].unique()), default=sorted(df["method"].unique())
    )
    sel_evals = st.multiselect(
        "Evaluations", sorted(df["eval"].unique()), default=sorted(df["eval"].unique())
    )

df = df[df["run"].isin(sel_runs) & df["method"].isin(sel_methods) & df["eval"].isin(sel_evals)]
if df.empty:
    st.info("Nothing selected.")
    st.stop()

use_log = st.sidebar.checkbox("Log y-axis", value=False)
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
