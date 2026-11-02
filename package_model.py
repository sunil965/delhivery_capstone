"""Package Model — Delhivery Delay Intelligence Pipeline.

Rebuilds the winning classifier pipeline from Part C, fits it on training rows,
exports model artifacts (model pickle, threshold, feature order, imputation values),
and verifies model reloading consistency on test data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import FEATURES, clean_shipments


def run_package_model():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    config_path = "outputs/part_c_final_config.csv"
    if not os.path.exists(config_path):
        raise FileNotFoundError("outputs/part_c_final_config.csv not found. Run part_c_classification.py first.")

    cfg = pd.read_csv(config_path).iloc[0]
    threshold = float(cfg["threshold"])
    n_estimators = int(cfg["n_estimators"])
    max_depth_val = cfg["max_depth"]
    max_depth = None if str(max_depth_val).lower() == "unlimited" or pd.isna(max_depth_val) else int(max_depth_val)
    min_samples_leaf = int(cfg["min_samples_leaf"])
    selector_k = int(cfg["selector_k"])

    # Load historical modelling table
    mod_df = pd.read_csv("data/modelling_table.csv")
    X = mod_df[FEATURES]
    y = mod_df["is_delayed"]

    # Rebuild Part C's stratified split: 70 / 15 / 15 -> 840 / 180 / 180
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Instantiate and fit the final Pipeline
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )),
    ])
    final_pipeline.fit(X_train, y_train)

    # 1. Save delay_model.pkl
    joblib.dump(final_pipeline, "models/delay_model.pkl")
    print("models/delay_model.pkl saved.")

    # 2. Save delay_threshold.csv
    pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
    print("models/delay_threshold.csv saved.")

    # 3. Save feature_order.csv
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    print("models/feature_order.csv saved.")

    # 4. Save impute_values.csv from clean_shipments on raw_shipments.csv
    raw_df = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw_df)
    impute_df = pd.DataFrame([
        {"column": "distance_km", "median": round(medians["distance_km"], 4)},
        {"column": "parcel_weight_kg", "median": round(medians["parcel_weight_kg"], 4)},
        {"column": "declared_value_inr", "median": round(medians["declared_value_inr"], 4)},
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)
    print("models/impute_values.csv saved.")

    # 5. Reload verification
    reloaded_pipeline = joblib.load("models/delay_model.pkl")
    probs_in_memory = final_pipeline.predict_proba(X_test)[:, 1]
    probs_reloaded = reloaded_pipeline.predict_proba(X_test)[:, 1]

    reload_max_abs_diff = float(np.max(np.abs(probs_reloaded - probs_in_memory)))

    reload_checks = [
        ("test_rows_scored", len(X_test)),
        ("reload_max_abs_probability_diff", reload_max_abs_diff),
        ("threshold", threshold),
    ]
    pd.DataFrame(reload_checks, columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)
    print("outputs/part_d_reload_check.csv written.")


if __name__ == "__main__":
    run_package_model()
