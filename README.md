DialectDrift

A prototype language-pair-level difficulty predictor: given a native language
and a target language, it estimates how linguistically distant they are and
turns that into a difficulty score. Built as a multi-page Streamlit app.

This is a resume/portfolio project, not a validated research tool.


What this is (and isn't)

- A transparent proxy model based on real linguistic distance data
- Predicts difficulty at the language-pair level (e.g. "English to Japanese")
- NOT a personalized per-learner model. It doesn't know anything about you
  individually (your age, other languages you speak, learning style, etc.)
- NOT comparable to Duolingo's SLAM (Second Language Acquisition Modeling)
  research, which uses millions of real learners' exercise-response logs to
  predict individual word/error-level difficulty. This project uses no
  learner data at all. It's a linguistics-features-only proxy.


App structure

The app is a multi-page Streamlit app:

- Home - overview and navigation
- Predictor - pick a native and target language, get a difficulty score
  and plain-language explanation
- Compare Languages - detailed feature-by-feature breakdown for any pair,
  in both directions
- Rankings - easiest/hardest global pairs, and easiest-next-language
  rankings for a chosen native language
- Model Insights - model comparison, selection rationale, and permutation
  feature importance
- About - full methodology, data sources, and limitations (mirrors this
  README)


Data source

All linguistic features come from URIEL / lang2vec:

Littell, P., Mortensen, D. R., Lin, K., Kairis, K., Turner, C., & Levin, L. (2017).
URIEL and lang2vec: Representing languages as typological, geographical, and
phylogenetic vectors. EACL 2017.

Distributed via the lang2vec pip package, which bundles real feature data
aggregated from WALS, Ethnologue, PHOIBLE, and Glottolog. No data was scraped,
simulated, or fabricated.

One feature, writing script similarity, is not available from URIEL and is
instead a small, manually compiled reference table (src/script_lookup.py)
based on public knowledge of which script each language uses. This is
explicitly a hand-built lookup, not a claimed dataset.

Known gap: true lexical (word-form) distance, as computed from ASJP wordlists
via edit-distance, was out of scope for this project. It requires a full
alignment pipeline over raw phonetic transcriptions and is not included. No
substitute number was invented in its place.

Also excluded: the raw syntax_wals and phonology_wals feature sets were
pulled and tested but dropped from modeling, since some languages had up to
100 percent missing values in these sets (verified during development). The
k-NN imputed versions (syntax_knn, phonology_knn, inventory_knn) were used
instead, since they have zero missing values across all 40 languages.

The "learned" embedding feature set (Malaviya et al. 2017), also part of
lang2vec, was tested and found to fail to load under current numpy versions
due to a pickle-loading restriction. It was excluded rather than worked
around silently.


Methodology

1. src/data_pull.py pulls 5 real feature vectors per language from URIEL
   for 40 languages: geographic, family/phylogenetic, syntactic,
   phonological, and phonological inventory.

2. src/feature_engineering.py builds a feature table of all 1,560 ordered
   language pairs (40 x 39), computing:
   - Euclidean distance for each of the 5 core feature types
   - Cosine distance for each of the 5 core feature types (captures
     directional/shape similarity, distinct from Euclidean magnitude)
   - A PCA-reduced combined typological distance (10 components,
     explaining about 66 percent of variance across all combined features)
   - A KMeans-based cluster grouping (6 clusters) and same_cluster flag
   - native_cluster_size, a rough linguistic-neighborhood-size proxy
   - same_script, from the manual writing-system lookup

   These are combined into a transparent, hand-weighted difficulty score,
   using only 5 of the raw Euclidean distances (kept deliberately simple
   and interpretable):

   difficulty = 0.30 * family_distance
              + 0.25 * phonological_distance
              + 0.20 * syntactic_distance
              + 0.15 * inventory_distance
              + 0.05 * geographic_distance
              + 0.05 * (1 - same_script)

   All distances are min-max normalized to [0, 1] first. These weights are
   a subjective design choice, loosely informed by L2-acquisition literature
   suggesting family and phonological distance are the strongest drivers of
   perceived difficulty. They are not fit to any ground-truth outcome data,
   because no such per-pair ground truth dataset was used.

3. src/train_model.py compares three models using 5-fold cross-validation:
   Linear Regression, Random Forest, and Gradient Boosting.


Results, reported honestly

Model comparison (5-fold cross-validation):

Linear Regression: CV RMSE ~0.0000, CV R-squared ~1.0000
Random Forest: CV RMSE ~0.0178, CV R-squared ~0.9722
Gradient Boosting: CV RMSE ~0.0095, CV R-squared ~0.9920

Important: Linear Regression's near-perfect score is an algebraic artifact,
not a genuine modeling result. difficulty_score is DEFINED as a linear
combination of a subset of the input features, so Linear Regression is
simply solving for coefficients we already wrote by hand. Selecting it as
"best" would be misleading. Gradient Boosting was deliberately selected
instead, since as a non-linear model it cannot trivially invert the formula,
and its feature importances are meaningfully informative rather than
trivially concentrated on the 5 formula inputs.

Final held-out test performance (Gradient Boosting):
Test RMSE: 0.0091
Test R-squared: 0.9930

Permutation importance (mean decrease in R-squared when a feature is
shuffled, more robust to feature correlation than a tree model's built-in
importance):

Overall typological distance (PCA): 0.227
Family distance: 0.132
Phonology distance (cosine): 0.132
Phonology distance: 0.085
Same writing script: 0.066
Syntax distance: 0.029
Sound inventory distance: 0.025
Syntax distance (cosine): 0.015
Sound inventory distance (cosine): 0.003
Family distance (cosine): 0.002
Geographic distance: 0.001
Geographic distance (cosine): 0.000
Same typological cluster: 0.000
Native's cluster size: 0.000

Honest negative result: the KMeans clustering features (same_cluster,
native_cluster_size) showed zero measured importance. They were kept in the
pipeline and disclosed rather than quietly removed, since their lack of
added value is itself informative (their signal likely overlaps with family
distance).


Other known limitations

- Symmetric scores. difficulty(A to B) equals difficulty(B to A), since all
  features are symmetric distances. In reality, difficulty is often
  asymmetric. Modeling that would require actual learner-outcome data.
- Small language set. Only 40 commonly studied languages are included, not
  the full set of roughly 7,000 languages URIEL covers.
- No personalization. Doesn't account for a learner's other known languages,
  age, motivation, or learning method.


Project structure

app/
  Home.py
  common.py
  pages/
    1_Predictor.py
    2_Compare_Languages.py
    3_Rankings.py
    4_Model_Insights.py
    5_About.py
data/
  raw/
  processed/
models/
src/
  data_pull.py
  feature_engineering.py
  train_model.py
  script_lookup.py
.streamlit/
  config.toml


Running it

pip install -r requirements.txt
python src/data_pull.py
python src/feature_engineering.py
python src/train_model.py
streamlit run app/Home.py


Tech stack

Python, pandas, scikit-learn, lang2vec (URIEL), Streamlit.