"""
DialectDrift — Language Learning Difficulty Predictor
Step 4: Streamlit app.

Prototype/proxy model — NOT personalized per-learner prediction.
See README.md for full methodology and limitations.
"""

import streamlit as st
import pandas as pd
import pickle
import json

DATA_PATH = "data/processed/language_pairs_features.csv"
MODEL_PATH = "models/difficulty_model.pkl"
METRICS_PATH = "models/metrics.json"

FEATURES = [
    "geographic_distance",
    "family_distance",
    "phonological_distance",
    "syntactic_distance",
    "inventory_distance",
    "same_script",
]

LANG_NAMES = {
    "eng": "English", "spa": "Spanish", "fra": "French", "deu": "German",
    "ita": "Italian", "por": "Portuguese", "nld": "Dutch", "rus": "Russian",
    "pol": "Polish", "cmn": "Mandarin Chinese", "jpn": "Japanese",
    "kor": "Korean", "arb": "Arabic", "hin": "Hindi", "tur": "Turkish",
    "vie": "Vietnamese", "tha": "Thai", "swh": "Swahili", "fin": "Finnish",
    "hun": "Hungarian", "ell": "Greek", "heb": "Hebrew",
    "ind": "Indonesian", "tgl": "Tagalog",
}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def difficulty_label(score):
    if score < 0.35:
        return "Easy"
    elif score < 0.55:
        return "Moderate"
    elif score < 0.7:
        return "Hard"
    else:
        return "Very Hard"


def explain(row):
    parts = []
    if row["family_distance"] > row["family_distance"].__class__(0.5) if False else None:
        pass
    return parts


def build_explanation(row):
    """Plain-language explanation driven by which normalized features are largest."""
    contribs = {
        "different language family": row["family_distance_norm"] * 0.30,
        "different sound system (phonology)": row["phonological_distance_norm"] * 0.25,
        "different sentence structure (syntax)": row["syntactic_distance_norm"] * 0.20,
        "different sound inventory": row["inventory_distance_norm"] * 0.15,
        "geographic/cultural distance": row["geographic_distance_norm"] * 0.05,
        "different writing script": (1 - row["same_script"]) * 0.05,
    }
    top = sorted(contribs.items(), key=lambda x: -x[1])[:2]
    return top


st.set_page_config(page_title="DialectDrift", page_icon="🗺️", layout="centered")

st.title("🗺️ DialectDrift")
st.caption("A linguistic-distance-based language learning difficulty predictor — prototype, not a personalized model.")

df = load_data()
model = load_model()
metrics = load_metrics()

languages = sorted(df["native"].unique())
lang_options = {LANG_NAMES.get(l, l): l for l in languages}

col1, col2 = st.columns(2)
with col1:
    native_display = st.selectbox("Your native language", sorted(lang_options.keys()), index=sorted(lang_options.keys()).index("English") if "English" in lang_options else 0)
with col2:
    target_display = st.selectbox("Language you want to learn", sorted(lang_options.keys()), index=1)

native = lang_options[native_display]
target = lang_options[target_display]

if native == target:
    st.warning("Pick two different languages.")
else:
    row = df[(df["native"] == native) & (df["target"] == target)].iloc[0]
    score = row["difficulty_score"]
    label = difficulty_label(score)

    st.metric(f"Predicted difficulty: {native_display} → {target_display}", f"{score:.3f}", label)

    top_drivers = build_explanation(row)
    driver_text = " and ".join([d[0] for d in top_drivers])
    st.write(f"**Why:** This score is driven mainly by **{driver_text}** between {native_display} and {target_display}.")

    with st.expander("See raw feature values"):
        st.dataframe(row[FEATURES].to_frame(name="value"))

    st.subheader("What drives difficulty scores in general")
    importances = metrics["feature_importances"]
    imp_df = pd.DataFrame(list(importances.items()), columns=["feature", "importance"]).sort_values("importance")
    st.bar_chart(imp_df.set_index("feature"))

    st.caption(
        f"Model: Random Forest Regressor · Test RMSE: {metrics['rmse']:.4f} · "
        f"Test R²: {metrics['r2']:.4f} (see README for what this R² does and doesn't mean)"
    )

    st.subheader(f"Easiest next languages for a {native_display} speaker")
    ranked = df[df["native"] == native].copy()
    ranked["target_name"] = ranked["target"].map(lambda t: LANG_NAMES.get(t, t))
    ranked = ranked.sort_values("difficulty_score")[["target_name", "difficulty_score"]].head(8)
    ranked.columns = ["Target language", "Difficulty score"]
    st.dataframe(ranked, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "⚠️ This is a language-pair-level prototype using linguistic distance proxies "
    "(URIEL/lang2vec), not a personalized per-learner model like Duolingo's SLAM. "
    "See README.md for full methodology and limitations."
)