"""
Step 1: Pull real linguistic distance data from URIEL/lang2vec.

Source: Littell, Mortensen, Lin, Kairis, Turner & Levin (2017),
"URIEL and lang2vec: Representing languages as typological,
geographical, and phylogenetic vectors," EACL 2017.
Distributed via the `lang2vec` pip package (bundled data, no external download).
"""

import lang2vec.lang2vec as l2v
import pandas as pd
import numpy as np

# A manageable, resume-project-sized set of commonly studied languages
# (ISO 639-3 codes, as required by lang2vec)
LANGUAGES = [
    "eng", "spa", "fra", "deu", "ita", "por", "nld",
    "rus", "pol", "cmn", "jpn", "kor", "arb", "hin",
    "tur", "vie", "tha", "swh", "fin", "hun", "ell",
    "heb", "ind", "tgl"
]

FEATURE_SETS = ["geo", "fam", "phonology_knn", "syntax_knn", "inventory_knn"]

def pull_raw_features():
    """Pull raw URIEL feature vectors for each language and feature set."""
    all_rows = []
    for fs in FEATURE_SETS:
        feats = l2v.get_features(LANGUAGES, fs)
        for lang, vec in feats.items():
            all_rows.append({
                "language": lang,
                "feature_set": fs,
                "vector": ",".join(str(x) for x in vec)
            })
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    df = pull_raw_features()
    df.to_csv("data/raw/uriel_raw_features.csv", index=False)
    print(f"Pulled {len(df)} rows across {len(FEATURE_SETS)} feature sets "
          f"for {len(LANGUAGES)} languages.")
    print(df.head())