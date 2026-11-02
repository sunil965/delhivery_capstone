import joblib
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

def run_scoring_pipeline():
    # Load production parameters safely
    new_shipments = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    model = joblib.load("models/delay_model.pkl")
    hub_segments = pd.read_csv("outputs/hub_segments.csv")

    # Process un-flagged entries matching training footprints exactly
    clean_df, _ = clean_shipments(new_shipments, medians=medians)
    featured_df = build_features(clean_df, hubs)

    # Impose structural ordering matrices
    X_prod = featured_df[feature_order]

    # Evaluate distribution metrics
    probs = model.predict_proba(X_prod)[:, 1]
    featured_df["delay_probability"] = probs.round(4)
    featured_df["flagged"] = (probs >= threshold).astype(int)

    # Connect hub structural configurations
    scored_df = featured_df.merge(hub_segments[["hub_id", "segment"]], on="hub_id", how="left")

    # Evaluate rule mapping matrices
    actions = []
    for _, row in scored_df.iterrows():
        if row["flagged"] == 1:
            if row["segment"] == "Escalate":
                actions.append("Escalate to hub manager")
            else:
                actions.append("Proactive customer call")
        else:
            actions.append("No action")

    scored_df["action"] = actions

    # Format according to delivery footprint criteria
    output_cols = ["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]
    scored_df[output_cols].to_csv("outputs/new_shipment_actions.csv", index=False)

if __name__ == "__main__":
    run_scoring_pipeline()
