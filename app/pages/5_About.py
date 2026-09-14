"""
DialectDrift — About page.
Full methodology, data sources, and honest limitations.
"""

import streamlit as st

st.set_page_config(page_title="DialectDrift — About", page_icon="🗺️", layout="centered")
from common import apply_theme
apply_theme()
st.title("About DialectDrift")

st.markdown("""
DialectDrift is a prototype language-pair-level difficulty predictor: given a
native language and a target language, it estimates how linguistically
distant they are and turns that into a difficulty score.

**This is a resume/portfolio project, not a validated research tool.**
""")

st.subheader("What this is (and isn't)")
st.markdown("""
- A transparent proxy model based on real linguistic distance data
- Predicts difficulty at the language-pair level (e.g. "English to Japanese")
- NOT a personalized per-learner model. It doesn't know anything about you
  individually (your age, other languages you speak, learning style, etc.)
- NOT comparable to Duolingo's SLAM (Second Language Acquisition Modeling)
  research, which uses millions of real learners' exercise-response logs to
  predict individual word/error-level difficulty. This project uses no
  learner data at all — it's a linguistics-features-only proxy.
""")

st.subheader("Data source")
st.markdown("""
All linguistic features come from **URIEL / lang2vec**:

> Littell, P., Mortensen, D. R., Lin, K., Kairis, K., Turner, C., & Levin, L. (2017).
> *URIEL and lang2vec: Representing languages as typological, geographical, and
> phylogenetic vectors.* EACL 2017.

Distributed via the `lang2vec` pip package, which bundles real feature data
aggregated from WALS, Ethnologue, PHOIBLE, and Glottolog. No data was scraped,
simulated, or fabricated.

One feature — **writing script similarity** — is not available from URIEL and
is instead a small, manually compiled reference table based on public
knowledge of which script each language uses. This is explicitly a
hand-built lookup, not a claimed dataset.

**Known gap:** true lexical (word-form) distance, as computed from ASJP
wordlists via edit-distance, was out of scope for this project. It requires a
full alignment pipeline over raw phonetic transcriptions and is not included.

**Also excluded:** the raw `syntax_wals` and `phonology_wals` feature sets
were tested but dropped — some languages had up to 100% missing values in
these sets. The k-NN imputed versions (`syntax_knn`, `phonology_knn`,
`inventory_knn`) were used instead, since they have zero missing values.
""")

st.subheader("Methodology")
st.markdown("""
1. Pull 5 real feature vectors per language from URIEL (geographic, family,
   syntactic, phonological, phonological inventory).
2. Compute both Euclidean and cosine distances between every language pair
   for each feature type, plus a PCA-reduced combined distance and
   KMeans-based cluster grouping.
3. Combine 5 of the raw Euclidean distances into a transparent, hand-weighted
   difficulty score:
""")

st.code(
    "difficulty = 0.30 * family_distance\n"
    "           + 0.25 * phonological_distance\n"
    "           + 0.20 * syntactic_distance\n"
    "           + 0.15 * inventory_distance\n"
    "           + 0.05 * geographic_distance\n"
    "           + 0.05 * (1 - same_script)",
    language="text"
)

st.markdown("""
All distances are min-max normalized to [0, 1] first. **These weights are a
subjective design choice**, loosely informed by L2-acquisition literature
suggesting family and phonological distance are the strongest drivers of
perceived difficulty — they are not fit to any ground-truth outcome data,
because no such per-pair ground truth dataset was used.

4. Train and compare three models (Linear Regression, Random Forest,
   Gradient Boosting) using 5-fold cross-validation.
""")

st.warning(
    "Linear Regression scores a near-perfect R² in this comparison. This is "
    "an algebraic artifact, not a real modeling result — the difficulty score "
    "is *defined* as a linear combination of a subset of the input features, "
    "so Linear Regression just solves for the coefficients we already wrote. "
    "A non-linear model (Gradient Boosting) was deliberately selected instead."
)

st.subheader("Other known limitations")
st.markdown("""
- **Symmetric scores.** difficulty(A to B) equals difficulty(B to A), since
  all features are symmetric distances. In reality, difficulty is often
  asymmetric (e.g. Portuguese speakers tend to understand Spanish more easily
  than the reverse). Modeling that would require actual learner-outcome data.
- **Small language set.** Only 40 commonly studied languages are included,
  not the full set of roughly 7,000 languages URIEL covers.
- **No personalization.** Doesn't account for a learner's other known
  languages, age, motivation, or learning method — all of which matter more
  in practice than pure linguistic distance.
- **Clustering features added no measured predictive value** in permutation
  importance testing, most likely because their signal overlaps with the
  language family distance feature. Kept in the pipeline and disclosed
  rather than quietly removed.
""")

st.subheader("Tech stack")
st.markdown("Python, pandas, scikit-learn, lang2vec (URIEL), Streamlit.")