"""
Step 3: Train a Random Forest regressor to predict the difficulty score
from the linguistic distance features. Honest train/test split + metrics.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "data/processed/language_pairs_features.csv"
MODEL_PATH = "models/difficulty_model.pkl"
METRICS_PATH = "models/metrics.json"

FEATURES = [
    "geographic_distance",
    "family_distance",
    "phonological_distance",
    "syntactic_distance",
    "inventory_distance",
    "same_script",
]
TARGET = "difficulty_score"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    print("=== Honest results, no inflation ===")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R^2:  {r2:.4f}")
    print("\nFeature importances:")
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")

    # Important caveat, printed explicitly so it's never hidden:
    print("\nNOTE: The target (difficulty_score) was itself constructed as a "
          "weighted formula of these same features (see feature_engineering.py). "
          "So a high R^2 here mostly reflects the model re-deriving that formula "
          "from data, NOT validated real-world learner difficulty. This is "
          "expected and is disclosed in the README as a known limitation of "
          "this prototype.")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "feature_importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()