"""
DialectDrift — Home page.
This file is the multi-page app's entry point; the pages/ folder next to
it is auto-detected by Streamlit and shown as sidebar navigation.
"""

import streamlit as st
from common import load_data, load_metrics

st.set_page_config(page_title="DialectDrift", page_icon="🗺️", layout="centered")
from common import apply_theme
apply_theme()
st.title("DialectDrift")
st.caption("A linguistic-distance-based language learning difficulty predictor")

df = load_data()
metrics = load_metrics()

st.write(
    "DialectDrift estimates how difficult it is to learn a target language "
    "given your native language, using real published linguistic distance "
    "data (URIEL/lang2vec) rather than guesswork."
)

col1, col2, col3 = st.columns(3)
col1.metric("Languages covered", df["native"].nunique())
col2.metric("Language pairs", len(df))
col3.metric("Model", metrics["best_model"])

st.divider()
st.subheader("What you can do here")
st.markdown(
    "- **Predictor** — pick a native and target language, get a difficulty "
    "score with a plain-language explanation\n"
    "- **Compare Languages** — see a detailed side-by-side feature breakdown "
    "for any pair\n"
    "- **Rankings** — browse the easiest and hardest language pairs, or the "
    "easiest next languages for a given native language\n"
    "- **Model Insights** — how the model was built, which model was chosen "
    "and why, and what actually drives its predictions\n"
    "- **About** — data sources, methodology, and honest limitations"
)

st.divider()
st.caption(
    "This is a language-pair-level prototype using linguistic distance proxies, "
    "not a personalized per-learner model. See the About page for details."
)