import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.shipment_prep import clean_shipments, FEATURES

def run_package_model():
    os.makedirs("models", exist_ok=True)

    # Load raw records to fetch medians safely
    raw_hist = pd.read_csv("data/raw_shipments.csv")
    _, medians = clean_shipments(raw_hist)

    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df["is_delayed"]

    # Rebuild localized stratified datasets matching training criteria exactly
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y)

    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]

    params = {"n_estimators": int(config["n_estimators"])}
    if str(config["max_depth"]) != "unlimited":
        params["max_depth"] = int(config["max_depth"])
    if "min_samples_leaf" in config and pd.notna(config["min_samples_leaf"]):
        params["min_samples_leaf"] = int(config["min_samples_leaf"])

    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **params))
    ])
    final_pipeline.fit(X_train, y_train)

    # Serialize core operational pipeline components
    joblib.dump(final_pipeline, "models/delay_model.pkl")

    pd.DataFrame([[float(config["threshold"])]], columns=["threshold"]).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame(FEATURES, columns=["feature"]).to_csv("models/feature_order.csv", index=False)

    pd.DataFrame(list(medians.items()), columns=["column", "median"]).to_csv("models/impute_values.csv", index=False)

    # Verify serialization integrity checks
    reloaded_pipe = joblib.load("models/delay_model.pkl")
    prob_in_mem = final_pipeline.predict_proba(X_test)[:, 1]
    prob_reloaded = reloaded_pipe.predict_proba(X_test)[:, 1]
    max_diff = float(np.max(np.abs(prob_in_mem - prob_reloaded)))

    pd.DataFrame([
        ["test_rows_scored", len(X_test)],
        ["reload_max_abs_probability_diff", round(max_diff, 4)],
        ["threshold", float(config["threshold"])]
    ], columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)

if __name__ == "__main__":
    run_package_model()
