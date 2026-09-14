"""
DialectDrift — Model Insights page.
Model comparison, selection rationale, and feature importance.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_metrics

st.set_page_config(page_title="DialectDrift — Model Insights", page_icon="🗺️", layout="centered")
from common import apply_theme
apply_theme()
st.title("Model Insights")

metrics = load_metrics()

st.subheader("Model comparison (5-fold cross-validation)")
comparison = metrics["model_comparison"]
comp_df = pd.DataFrame([
    {"Model": name, "CV RMSE": vals["cv_rmse"], "CV R²": vals["cv_r2"]}
    for name, vals in comparison.items()
])
st.dataframe(comp_df, hide_index=True, use_container_width=True)

st.warning(metrics.get("linear_regression_caveat", ""))

st.subheader(f"Selected model: {metrics['best_model']}")
col1, col2 = st.columns(2)
col1.metric("Held-out test RMSE", f"{metrics['test_rmse']:.4f}")
col2.metric("Held-out test R²", f"{metrics['test_r2']:.4f}")

st.caption(
    "Selected deliberately over Linear Regression despite Linear Regression's "
    "higher raw R², because that R² is an algebraic artifact rather than a "
    "genuine result — see the warning above."
)

st.divider()
st.subheader("What actually drives predictions (permutation importance)")
st.caption(
    "Permutation importance measures how much the model's accuracy drops "
    "when a feature's values are randomly shuffled — a more reliable signal "
    "than a tree model's built-in importance, which can be skewed by "
    "correlated features."
)

readable_names = {
    "geo_euclidean": "Geographic distance",
    "geo_cosine": "Geographic distance (cosine)",
    "fam_euclidean": "Family distance",
    "fam_cosine": "Family distance (cosine)",
    "syntax_knn_euclidean": "Syntax distance",
    "syntax_knn_cosine": "Syntax distance (cosine)",
    "phonology_knn_euclidean": "Phonology distance",
    "phonology_knn_cosine": "Phonology distance (cosine)",
    "inventory_knn_euclidean": "Sound inventory distance",
    "inventory_knn_cosine": "Sound inventory distance (cosine)",
    "pca_combined_distance": "Overall typological distance (PCA)",
    "same_cluster": "Same typological cluster",
    "native_cluster_size": "Native's cluster size",
    "same_script": "Same writing script",
}

imp = metrics["feature_importances"]
imp_df = pd.DataFrame([
    {"Feature": readable_names.get(f, f), "Importance": v} for f, v in imp.items()
]).sort_values("Importance")

fig = go.Figure(go.Bar(
    x=imp_df["Importance"],
    y=imp_df["Feature"],
    orientation="h",
    marker=dict(
        color=imp_df["Importance"],
        colorscale=[[0, "#1B2A38"], [1, "#2A6F77"]],
        line=dict(color="#2A6F77", width=1),
    ),
))
fig.update_layout(
    plot_bgcolor="#101820",
    paper_bgcolor="#101820",
    font=dict(color="#EDEDED"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=420,
    xaxis=dict(gridcolor="#1B2A38", title="Permutation importance"),
    yaxis=dict(title=""),
)
st.plotly_chart(fig, use_container_width=True)

zero_importance = [readable_names.get(f, f) for f, v in imp.items() if v == 0]
if zero_importance:
    st.info(
        f"These engineered features showed zero measured importance: "
        f"{', '.join(zero_importance)}. This is an honest negative result — "
        "they didn't add predictive value beyond what the other features "
        "already captured, most likely because clustering-based structure "
        "overlaps heavily with the language family distance feature."
    )

st.divider()
st.caption(
    f"Training set size: {metrics['n_train']} pairs · Test set size: {metrics['n_test']} pairs"
)