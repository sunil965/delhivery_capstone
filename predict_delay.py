import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features


def run_predict_delay():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load inputs
    new_shipments_path = "data/new_shipments.csv"
    hubs_path = "data/hubs.csv"
    model_path = "models/delay_model.pkl"
    threshold_path = "models/delay_threshold.csv"
    feature_order_path = "models/feature_order.csv"
    impute_path = "models/impute_values.csv"
    segments_path = "outputs/hub_segments.csv"

    for p in [new_shipments_path, hubs_path, model_path, threshold_path,
              feature_order_path, impute_path, segments_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file {p} does not exist. Run preceding pipeline scripts.")

    new_raw = pd.read_csv(new_shipments_path)
    hubs_df = pd.read_csv(hubs_path)
    model = joblib.load(model_path)
    threshold = float(pd.read_csv(threshold_path)["threshold"].iloc[0])
    feature_order = pd.read_csv(feature_order_path)["feature"].tolist()
    impute_df = pd.read_csv(impute_path)
    stored_medians = dict(zip(impute_df["column"], impute_df["median"].astype(float)))
    hub_segments_df = pd.read_csv(segments_path)

    # 2. Clean new shipments using stored medians
    # Note: Preserve original order
    clean_new, _ = clean_shipments(new_raw, medians=stored_medians)

    # 3. Attach features using build_features
    featured_new = build_features(clean_new, hubs_df)

    # 4. Align features strictly to feature_order
    X_new = featured_new[feature_order]

    # 5. Predict probabilities (positive class, rounded to 4 decimals)
    delay_probs = np.round(model.predict_proba(X_new)[:, 1], 4)
    clean_new["delay_probability"] = delay_probs
    clean_new["flagged"] = (clean_new["delay_probability"] >= threshold).astype(int)

    # 6. Join hub segment from hub_segments.csv
    scored_df = clean_new.merge(
        hub_segments_df[["hub_id", "segment"]], on="hub_id", how="left"
    )

    # 7. Assign operational action
    def determine_action(row):
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                return "Escalate to hub manager"
            return "Proactive customer call"
        return "No action"

    scored_df["action"] = scored_df.apply(determine_action, axis=1)

    # 8. Output new_shipment_actions.csv (shipment_id, hub_id, segment, delay_probability, flagged, action)
    output_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    final_output = scored_df[output_cols]
    final_output.to_csv("outputs/new_shipment_actions.csv", index=False)
    print(f"outputs/new_shipment_actions.csv written with {len(final_output)} records.")


if __name__ == "__main__":
    run_predict_delay()