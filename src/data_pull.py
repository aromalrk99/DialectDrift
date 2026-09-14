"""
Step 1 (v2): Pull an expanded set of real linguistic feature vectors from URIEL/lang2vec.

Source: Littell, Mortensen, Lin, Kairis, Turner & Levin (2017),
"URIEL and lang2vec: Representing languages as typological,
geographical, and phylogenetic vectors," EACL 2017.
Distributed via the `lang2vec` pip package (bundled data, no external download).

NOTE: the "learned" embedding feature set (Malaviya et al. 2017) is also part
of lang2vec but fails to load under current numpy versions (pickle-loading
restriction). It is intentionally excluded rather than worked around silently.
"""

import lang2vec.lang2vec as l2v
import pandas as pd

# Expanded to 40 languages, still all real ISO 639-3 codes with full URIEL coverage
LANGUAGES = [
    "eng", "spa", "fra", "deu", "ita", "por", "nld",
    "rus", "pol", "cmn", "jpn", "kor", "arb", "hin",
    "tur", "vie", "tha", "swh", "fin", "hun", "ell",
    "heb", "ind", "tgl",
    "ces", "swe", "nor", "dan", "ron", "ukr",
    "ben", "urd", "pan", "tam", "tel", "mar",
    "jav", "mya", "khm", "amh",
]

# Reverted to _knn (fully k-NN imputed, zero missing values) after testing showed
# _average sets still have real gaps (930/443/316 missing entries across our
# languages) and raw _wals sets are unusable (up to 100% missing for some
# languages, e.g. phonology_wals). This is a correction from an earlier draft.
FEATURE_SETS = [
    "geo",              # geographic distance proxy (URIEL)
    "fam",              # phylogenetic family membership (Glottolog)
    "syntax_knn",       # syntax features, kNN-imputed, zero missing
    "phonology_knn",    # phonology features, kNN-imputed, zero missing
    "inventory_knn",    # phonological inventory features, kNN-imputed, zero missing
]


def pull_raw_features():
    all_rows = []
    for fs in FEATURE_SETS:
        feats = l2v.get_features(LANGUAGES, fs)
        for lang, vec in feats.items():
            all_rows.append({
                "language": lang,
                "feature_set": fs,
                "vector": ",".join(str(x) for x in vec),
            })
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = pull_raw_features()
    df.to_csv("data/raw/uriel_raw_features.csv", index=False)
    print(f"Pulled {len(df)} rows across {len(FEATURE_SETS)} feature sets "
          f"for {len(LANGUAGES)} languages.")
    print(df.groupby("feature_set")["vector"].count())