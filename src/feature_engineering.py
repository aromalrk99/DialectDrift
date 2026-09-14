"""
Step 2 (v2): Advanced feature engineering.

Adds, on top of v1's Euclidean distances:
- Cosine distance for each core feature type (direction/shape similarity,
  a genuinely different signal from Euclidean magnitude distance)
- PCA-based combined typological distance (dimensionality reduction across
  all feature types into one joint space)
- KMeans-based language clustering -> same_cluster feature (data-driven
  grouping, distinct from the raw Glottolog family-membership feature)
- native_cluster_size (how many of our languages share the native language's
  cluster) as a rough "linguistic neighborhood size" proxy
"""

import pandas as pd
import numpy as np
from itertools import permutations
from scipy.spatial.distance import euclidean, cosine
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from script_lookup import same_script

RAW_PATH = "data/raw/uriel_raw_features.csv"
OUT_PATH = "data/processed/language_pairs_features.csv"

CORE_SETS = ["geo", "fam", "syntax_knn", "phonology_knn", "inventory_knn"]
N_CLUSTERS = 6
N_PCA_COMPONENTS = 10


def load_raw_vectors():
    df = pd.read_csv(RAW_PATH)
    vectors = {}
    for fs in df["feature_set"].unique():
        sub = df[df["feature_set"] == fs]
        vectors[fs] = {
            row["language"]: np.array([float(x) for x in row["vector"].split(",")])
            for _, row in sub.iterrows()
        }
    return vectors


def build_combined_matrix(vectors, languages):
    """Concatenate all core feature vectors per language into one big vector."""
    combined = {
        lang: np.concatenate([vectors[fs][lang] for fs in CORE_SETS])
        for lang in languages
    }
    X = np.array([combined[l] for l in languages])
    return combined, X


def fit_pca_and_clusters(X, languages):
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X)
    pca_vectors = {lang: X_pca[i] for i, lang in enumerate(languages)}

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)
    clusters = {lang: int(cluster_labels[i]) for i, lang in enumerate(languages)}

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: {N_PCA_COMPONENTS} components explain "
          f"{explained_var:.1%} of variance in the combined feature space.")

    return pca_vectors, clusters


def build_pair_table(vectors, pca_vectors, clusters):
    languages = list(vectors["geo"].keys())
    cluster_sizes = pd.Series(clusters).value_counts().to_dict()
    rows = []

    for native, target in permutations(languages, 2):
        row = {"native": native, "target": target}

        # Euclidean + cosine distance per core feature type
        for fs in CORE_SETS:
            v1, v2 = vectors[fs][native], vectors[fs][target]
            row[f"{fs}_euclidean"] = euclidean(v1, v2)
            row[f"{fs}_cosine"] = cosine(v1, v2)

        # PCA-based combined typological distance
        row["pca_combined_distance"] = euclidean(pca_vectors[native], pca_vectors[target])

        # Cluster-based features
        row["same_cluster"] = int(clusters[native] == clusters[target])
        row["native_cluster_size"] = cluster_sizes[clusters[native]]

        # Script similarity (manual lookup)
        row["same_script"] = same_script(native, target)

        rows.append(row)

    return pd.DataFrame(rows)


def normalize(series):
    return (series - series.min()) / (series.max() - series.min())


def add_difficulty_score(df):
    """
    Transparent composite difficulty score, 0 (easiest) to 1 (hardest).
    Formula unchanged in spirit from v1 -- still built ONLY from the core
    Euclidean distances, so it stays interpretable. The new cosine/PCA/
    cluster features are extra model inputs, not part of the target formula
    itself, which keeps them from just re-deriving the label trivially.

        difficulty = 0.30 * fam_euclidean_norm
                   + 0.25 * phonology_knn_euclidean_norm
                   + 0.20 * syntax_knn_euclidean_norm
                   + 0.15 * inventory_knn_euclidean_norm
                   + 0.05 * geo_euclidean_norm
                   + 0.05 * (1 - same_script)

    Weights are a subjective design choice loosely informed by L2-acquisition
    literature (family and phonological distance as strongest difficulty
    drivers), NOT fit to any ground-truth learner outcome data.
    """
    for col in ["fam_euclidean", "phonology_knn_euclidean", "syntax_knn_euclidean",
                "inventory_knn_euclidean", "geo_euclidean"]:
        df[f"{col}_norm"] = normalize(df[col])

    df["difficulty_score"] = (
        0.30 * df["fam_euclidean_norm"] +
        0.25 * df["phonology_knn_euclidean_norm"] +
        0.20 * df["syntax_knn_euclidean_norm"] +
        0.15 * df["inventory_knn_euclidean_norm"] +
        0.05 * df["geo_euclidean_norm"] +
        0.05 * (1 - df["same_script"])
    )
    return df


if __name__ == "__main__":
    vectors = load_raw_vectors()
    languages = list(vectors["geo"].keys())

    combined, X = build_combined_matrix(vectors, languages)
    pca_vectors, clusters = fit_pca_and_clusters(X, languages)

    df = build_pair_table(vectors, pca_vectors, clusters)
    df = add_difficulty_score(df)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nBuilt {len(df)} language pairs with {df.shape[1]} columns.")
    print(f"Columns: {list(df.columns)}")
    print("\nEasiest pairs:")
    print(df[["native", "target", "difficulty_score"]].sort_values("difficulty_score").head(5))
    print("\nHardest pairs:")
    print(df[["native", "target", "difficulty_score"]].sort_values("difficulty_score").tail(5))