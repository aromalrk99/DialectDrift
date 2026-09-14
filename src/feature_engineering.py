"""
Step 2: Build a (native_language, target_language) feature table
and a transparent, clearly-labeled difficulty score.
"""

import pandas as pd
import numpy as np
from itertools import permutations
from scipy.spatial.distance import euclidean
import ast

from script_lookup import same_script

RAW_PATH = "data/raw/uriel_raw_features.csv"
OUT_PATH = "data/processed/language_pairs_features.csv"


def load_raw_vectors():
    """Reload the raw URIEL vectors and pivot into {feature_set: {lang: np.array}}."""
    df = pd.read_csv(RAW_PATH)
    vectors = {}
    for fs in df["feature_set"].unique():
        sub = df[df["feature_set"] == fs]
        vectors[fs] = {
            row["language"]: np.array([float(x) for x in row["vector"].split(",")])
            for _, row in sub.iterrows()
        }
    return vectors


def build_pair_table(vectors):
    languages = list(vectors["geo"].keys())
    rows = []

    for native, target in permutations(languages, 2):
        geo_dist = euclidean(vectors["geo"][native], vectors["geo"][target])
        fam_dist = euclidean(vectors["fam"][native], vectors["fam"][target])
        phon_dist = euclidean(vectors["phonology_knn"][native], vectors["phonology_knn"][target])
        syn_dist = euclidean(vectors["syntax_knn"][native], vectors["syntax_knn"][target])
        inv_dist = euclidean(vectors["inventory_knn"][native], vectors["inventory_knn"][target])
        script_match = same_script(native, target)

        rows.append({
            "native": native,
            "target": target,
            "geographic_distance": geo_dist,
            "family_distance": fam_dist,
            "phonological_distance": phon_dist,
            "syntactic_distance": syn_dist,
            "inventory_distance": inv_dist,
            "same_script": script_match,
        })

    return pd.DataFrame(rows)


def normalize(series):
    """Min-max normalize a column to [0, 1] so features are comparable."""
    return (series - series.min()) / (series.max() - series.min())


def add_difficulty_score(df):
    """
    Transparent composite difficulty score, 0 (easiest) to 1 (hardest).

    Formula (weights are a subjective, clearly-labeled design choice —
    NOT derived from any ground-truth learner outcome data):

        difficulty = 0.30 * family_distance_norm
                   + 0.25 * phonological_distance_norm
                   + 0.20 * syntactic_distance_norm
                   + 0.15 * inventory_distance_norm
                   + 0.05 * geographic_distance_norm
                   + 0.05 * (1 - same_script)

    Rationale for weights: family and phonological distance are typically
    considered the strongest drivers of subjective learning difficulty in
    L2 acquisition literature (e.g. FSI language difficulty rankings),
    followed by syntax and sound inventory; geography and script are
    included as smaller adjustments since they correlate with but don't
    directly cause difficulty.
    """
    df["family_distance_norm"] = normalize(df["family_distance"])
    df["phonological_distance_norm"] = normalize(df["phonological_distance"])
    df["syntactic_distance_norm"] = normalize(df["syntactic_distance"])
    df["inventory_distance_norm"] = normalize(df["inventory_distance"])
    df["geographic_distance_norm"] = normalize(df["geographic_distance"])

    df["difficulty_score"] = (
        0.30 * df["family_distance_norm"] +
        0.25 * df["phonological_distance_norm"] +
        0.20 * df["syntactic_distance_norm"] +
        0.15 * df["inventory_distance_norm"] +
        0.05 * df["geographic_distance_norm"] +
        0.05 * (1 - df["same_script"])
    )
    return df


if __name__ == "__main__":
    vectors = load_raw_vectors()
    df = build_pair_table(vectors)
    df = add_difficulty_score(df)
    df.to_csv(OUT_PATH, index=False)
    print(f"Built {len(df)} language pairs.")
    print(df[["native", "target", "difficulty_score"]].sort_values("difficulty_score").head(10))
    print("...")
    print(df[["native", "target", "difficulty_score"]].sort_values("difficulty_score").tail(10))