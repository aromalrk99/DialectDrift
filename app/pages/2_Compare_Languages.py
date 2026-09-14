"""
DialectDrift — Compare Languages page.
Detailed side-by-side feature breakdown for any language pair.
"""

import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_data, difficulty_label, lang_display_map, FEATURES

st.set_page_config(page_title="DialectDrift — Compare", page_icon="🗺️", layout="centered")
from common import apply_theme
apply_theme()
st.title("Compare Languages")
st.caption("A detailed, feature-by-feature breakdown for any language pair, in both directions.")

df = load_data()
languages = sorted(df["native"].unique())
lang_options = lang_display_map(languages)
display_names = sorted(lang_options.keys())

col1, col2 = st.columns(2)
with col1:
    a_display = st.selectbox("Language A", display_names, index=display_names.index("English") if "English" in display_names else 0, key="cmp_a")
with col2:
    b_display = st.selectbox("Language B", display_names, index=display_names.index("Spanish") if "Spanish" in display_names else 1, key="cmp_b")

a = lang_options[a_display]
b = lang_options[b_display]

if a == b:
    st.warning("Pick two different languages.")
    st.stop()

row_ab = df[(df["native"] == a) & (df["target"] == b)].iloc[0]
row_ba = df[(df["native"] == b) & (df["target"] == a)].iloc[0]

col1, col2 = st.columns(2)
col1.metric(f"{a_display} → {b_display}", f"{row_ab['difficulty_score']:.3f}", difficulty_label(row_ab["difficulty_score"]))
col2.metric(f"{b_display} → {a_display}", f"{row_ba['difficulty_score']:.3f}", difficulty_label(row_ba["difficulty_score"]))

if abs(row_ab["difficulty_score"] - row_ba["difficulty_score"]) < 1e-9:
    st.info(
        "Both directions show the same score. This is expected and is a known "
        "limitation of the current model, not a coincidence for this pair specifically — "
        "see the About page for details on why difficulty isn't yet modeled as asymmetric."
    )

st.divider()
st.subheader("Feature-by-feature comparison")

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

compare_df = pd.DataFrame({
    "Feature": [readable_names.get(f, f) for f in FEATURES],
    "Value": [row_ab[f] for f in FEATURES],
})
st.dataframe(compare_df, hide_index=True, use_container_width=True)

st.caption(
    "Distance features are symmetric (same value regardless of direction), "
    "which is why only one row set is shown above."
)