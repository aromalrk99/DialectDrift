"""
DialectDrift — Rankings page.
Global easiest/hardest pairs, and easiest-next-languages for a given native language.
"""

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_data, lang_display_map

st.set_page_config(page_title="DialectDrift — Rankings", page_icon="🗺️", layout="centered")
from common import apply_theme
apply_theme()
st.title("Rankings")

df = load_data()
languages = sorted(df["native"].unique())
lang_options = lang_display_map(languages)
display_names = sorted(lang_options.keys())

st.subheader("Easiest next language to learn")
native_display = st.selectbox(
    "Your native language", display_names,
    index=display_names.index("English") if "English" in display_names else 0
)
native = lang_options[native_display]

ranked = df[df["native"] == native].copy()
ranked["Target language"] = ranked["target"].map(lambda t: lang_options and next((k for k, v in lang_options.items() if v == t), t))
ranked = ranked.sort_values("difficulty_score")[["Target language", "difficulty_score"]]
ranked.columns = ["Target language", "Difficulty score"]

n = st.slider("How many to show", 5, len(ranked), 10)
st.dataframe(ranked.head(n), hide_index=True, use_container_width=True)

st.divider()
st.subheader("Globally easiest language pairs")
global_easiest = df.copy()
global_easiest["Pair"] = global_easiest.apply(
    lambda r: f"{next((k for k, v in lang_options.items() if v == r['native']), r['native'])} → "
              f"{next((k for k, v in lang_options.items() if v == r['target']), r['target'])}",
    axis=1
)
easiest = global_easiest.sort_values("difficulty_score")[["Pair", "difficulty_score"]].head(10)
easiest.columns = ["Pair", "Difficulty score"]
st.dataframe(easiest, hide_index=True, use_container_width=True)

st.subheader("Globally hardest language pairs")
hardest = global_easiest.sort_values("difficulty_score", ascending=False)[["Pair", "difficulty_score"]].head(10)
hardest.columns = ["Pair", "Difficulty score"]
st.dataframe(hardest, hide_index=True, use_container_width=True)

st.caption(
    "Rankings reflect linguistic distance proxies, not validated learner outcomes. "
    "See the About page for full methodology and limitations."
)