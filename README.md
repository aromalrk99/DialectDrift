DialectDrift

A prototype language-pair-level difficulty predictor: given a native language
and a target language, it estimates how linguistically distant they are and
turns that into a "difficulty score."

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


Data source

All linguistic features come from URIEL / lang2vec:

Littell, P., Mortensen, D. R., Lin, K., Kairis, K., Turner, C., & Levin, L. (2017).
URIEL and lang2vec: Representing languages as typological, geographical, and
phylogenetic vectors. EACL 2017.

Distributed via the lang2vec pip package, which bundles real feature data
aggregated from WALS, Ethnologue, PHOIBLE, and Glottolog. No data was scraped,
simulated, or fabricated.

One feature, writing script similarity, is not available from URIEL and
is instead a small, manually compiled reference table (src/script_lookup.py)
based on public knowledge of which script each language uses. This is
explicitly a hand-built lookup, not a claimed dataset.

Known gap: true lexical (word-form) distance, as computed from ASJP
wordlists via edit-distance, was out of scope for this project's timeframe.
It requires a full alignment pipeline over raw phonetic transcriptions. It is
not included, and no substitute number was invented in its place.


Methodology

1. src/data_pull.py pulls 5 real feature vectors per language from
   URIEL (geographic, family/phylogenetic, phonological, syntactic, phonological
   inventory).

2. src/feature_engineering.py computes Euclidean distances between
   every language pair for each feature, plus a script-match flag, then
   combines them into a single transparent difficulty score:

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

3. src/train_model.py trains a Random Forest Regressor to predict
   the difficulty score from the raw (non-normalized) distance features.


Results, reported honestly

Test RMSE: 0.0173
Test R-squared: 0.9724

Important caveat: the R-squared is high mostly because the model is
re-learning a formula it was trained on (the difficulty score is a weighted
function of these same features), not because it's been validated against
real learner outcomes. No such validation data exists in this project. Treat
this model as "a Random Forest that has learned to approximate a hand-designed
formula," not as "a model proven to predict real-world learning difficulty."

Feature importances:

Phonological distance: 0.509
Family distance: 0.221
Geographic distance: 0.119
Syntactic distance: 0.087
Inventory distance: 0.049
Same script: 0.014

Note importances don't exactly mirror the formula's weights. Random Forest
importance reflects how useful a feature is for splitting data, which is
affected by correlation between features (e.g. phonological distance and
sound inventory distance are related).


Other known limitations

- Symmetric scores. Currently, difficulty(A to B) equals difficulty(B to A),
  since all features are symmetric distances. In reality, difficulty is often
  asymmetric (e.g. Portuguese speakers tend to understand Spanish more easily
  than the reverse). Modeling that would require actual learner-outcome data.
- Small language set. Only 24 commonly studied languages are included,
  not the full set of roughly 7,000 languages URIEL covers.
- No personalization. Doesn't account for a learner's other known
  languages, age, motivation, or learning method, all of which matter more
  in practice than pure linguistic distance.


Running it

pip install -r requirements.txt
python src/data_pull.py
python src/feature_engineering.py
python src/train_model.py
streamlit run app/streamlit_app.py


Tech stack

Python, pandas, scikit-learn, lang2vec (URIEL), Streamlit.