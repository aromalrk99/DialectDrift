"""
DialectDrift — Predictor page.
Pick a native and target language, get a difficulty score with explanation.
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_data, load_metrics, difficulty_label, lang_display_map, FEATURES

st.set_page_config(page_title="DialectDrift — Predictor", page_icon="🗺️", layout="centered")
from common import apply_theme
apply_theme()
st.title("Predictor")
st.caption("Pick a native language and a target language to learn.")

df = load_data()
metrics = load_metrics()

languages = sorted(df["native"].unique())
lang_options = lang_display_map(languages)
display_names = sorted(lang_options.keys())

col1, col2 = st.columns(2)
with col1:
    native_display = st.selectbox(
        "Your native language", display_names,
        index=display_names.index("English") if "English" in display_names else 0
    )
with col2:
    target_display = st.selectbox(
        "Language you want to learn", display_names,
        index=display_names.index("Japanese") if "Japanese" in display_names else 1
    )

native = lang_options[native_display]
target = lang_options[target_display]

if native == target:
    st.warning("Pick two different languages.")
    st.stop()

row = df[(df["native"] == native) & (df["target"] == target)].iloc[0]
score = row["difficulty_score"]
label = difficulty_label(score)

st.metric(f"{native_display} → {target_display}", f"{score:.3f}", label)

# Build explanation from the model's own permutation importances,
# weighted by how large each of this pair's actual distances are.
importances = metrics["feature_importances"]
readable_names = {
    "geo_euclidean": "geographic distance",
    "geo_cosine": "geographic distance (directional)",
    "fam_euclidean": "language family distance",
    "fam_cosine": "language family distance (directional)",
    "syntax_knn_euclidean": "sentence structure (syntax) distance",
    "syntax_knn_cosine": "sentence structure (syntax) distance (directional)",
    "phonology_knn_euclidean": "sound system (phonology) distance",
    "phonology_knn_cosine": "sound system (phonology) distance (directional)",
    "inventory_knn_euclidean": "sound inventory distance",
    "inventory_knn_cosine": "sound inventory distance (directional)",
    "pca_combined_distance": "overall typological distance",
    "same_cluster": "typological cluster grouping",
    "native_cluster_size": "linguistic neighborhood size",
    "same_script": "writing script similarity",
}

# Normalize this pair's own raw feature values to rank what stands out
pair_signal = {}
for feat in FEATURES:
    col_vals = df[feat]
    val = row[feat]
    if col_vals.max() > col_vals.min():
        norm_val = (val - col_vals.min()) / (col_vals.max() - col_vals.min())
    else:
        norm_val = 0
    pair_signal[feat] = norm_val * importances.get(feat, 0)

top_drivers = sorted(pair_signal.items(), key=lambda x: -x[1])[:2]
driver_text = " and ".join([readable_names.get(f, f) for f, _ in top_drivers])

st.write(f"**Why:** For this pair, the biggest contributors are **{driver_text}**.")

with st.expander("See all raw feature values for this pair"):
    st.dataframe(row[FEATURES].to_frame(name="value"))

st.divider()
st.caption(
    f"Model: {metrics['best_model']} · Test R²: {metrics['test_r2']:.3f} · "
    "See Model Insights and About pages for full context on what this score does and doesn't mean."
)