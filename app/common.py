"""
Shared data/model loaders and constants used across all DialectDrift pages.
"""

import streamlit as st
import pandas as pd
import pickle
import json

DATA_PATH = "data/processed/language_pairs_features.csv"
MODEL_PATH = "models/difficulty_model.pkl"
METRICS_PATH = "models/metrics.json"

FEATURES = [
    "geo_euclidean", "geo_cosine",
    "fam_euclidean", "fam_cosine",
    "syntax_knn_euclidean", "syntax_knn_cosine",
    "phonology_knn_euclidean", "phonology_knn_cosine",
    "inventory_knn_euclidean", "inventory_knn_cosine",
    "pca_combined_distance",
    "same_cluster",
    "native_cluster_size",
    "same_script",
]

LANG_NAMES = {
    "eng": "English", "spa": "Spanish", "fra": "French", "deu": "German",
    "ita": "Italian", "por": "Portuguese", "nld": "Dutch", "rus": "Russian",
    "pol": "Polish", "cmn": "Mandarin Chinese", "jpn": "Japanese",
    "kor": "Korean", "arb": "Arabic", "hin": "Hindi", "tur": "Turkish",
    "vie": "Vietnamese", "tha": "Thai", "swh": "Swahili", "fin": "Finnish",
    "hun": "Hungarian", "ell": "Greek", "heb": "Hebrew",
    "ind": "Indonesian", "tgl": "Tagalog", "ces": "Czech", "swe": "Swedish",
    "nor": "Norwegian", "dan": "Danish", "ron": "Romanian", "ukr": "Ukrainian",
    "ben": "Bengali", "urd": "Urdu", "pan": "Punjabi", "tam": "Tamil",
    "tel": "Telugu", "mar": "Marathi", "jav": "Javanese", "mya": "Burmese",
    "khm": "Khmer", "amh": "Amharic",
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


def lang_display_map(languages):
    """Maps display names -> ISO codes for selectbox use."""
    return {LANG_NAMES.get(l, l): l for l in languages}

def apply_theme():
    """Injects shared custom CSS so every page has a consistent, non-default look."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', serif !important;
        letter-spacing: 0.2px;
        color: #ECE6DA !important;
    }

    h1 {
        border-bottom: 1px solid #C98A3E;
        padding-bottom: 0.4rem;
    }

    hr {
        border: none;
        border-top: 1px solid #C98A3E;
        opacity: 0.5;
        margin: 1.6rem 0;
    }

    div[data-testid="stMetric"] {
        background-color: #17222C;
        border: 1px solid rgba(201, 138, 62, 0.35);
        border-radius: 4px;
        padding: 1rem 1.2rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #8B97A3 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #C98A3E !important;
        font-family: 'Source Serif 4', serif;
    }

    [data-testid="stSidebar"] {
        background-color: #17222C;
        border-right: 1px solid rgba(201, 138, 62, 0.2);
    }

    [data-testid="stSidebar"] a {
        color: #ECE6DA !important;
        border-radius: 3px;
    }

    [data-testid="stSidebar"] a:hover {
        color: #C98A3E !important;
    }

    .stDataFrame {
        border: 1px solid rgba(201, 138, 62, 0.25);
        border-radius: 4px;
    }

    .stButton button, .stSelectbox div[data-baseweb="select"] {
        border-radius: 3px !important;
        border-color: rgba(201, 138, 62, 0.4) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 4px;
        border-left: 3px solid #C98A3E;
    }
    </style>
    """, unsafe_allow_html=True)