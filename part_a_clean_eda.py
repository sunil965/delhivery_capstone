import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES

def run_part_a():
    os.makedirs("outputs", exist_ok=True)

    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # Metrics collection before clearing features
    raw_rows = len(raw)
    dup_rows_removed = raw_rows - len(raw.drop_duplicates())

    df_dedup = raw.drop_duplicates().copy()
    missing_dist = df_dedup["distance_km"].isna().sum()

    # Sentinel counting before removal
    sentinels = (df_dedup["parcel_weight_kg"] > 100).sum()
    missing_weight = df_dedup["parcel_weight_kg"].isna().sum()

    # Clean text numbers for accurate base counts
    df_dedup["declared_value_clean"] = df_dedup["declared_value"].astype(str).str.replace("₹","",regex=False).str.replace(",","",regex=False)
    df_dedup["declared_value_clean"] = pd.to_numeric(df_dedup["declared_value_clean"], errors='coerce')
    missing_val = df_dedup["declared_value_clean"].isna().sum()
    missing_pay = df_dedup["payment_mode"].isna().sum()
    spellings_before = df_dedup["service_type"].nunique()

    # Perform cleaning
    clean_df, medians = clean_shipments(raw)
    spellings_after = clean_df["service_type"].nunique()

    # Feature construction
    modelling_table = build_features(clean_df, hubs)

    # Targets setup
    modelling_table["is_delayed"] = (modelling_table["delivery_hours"] > modelling_table["promised_hours"]).astype(int)
    modelling_table.to_csv("data/modelling_table.csv", index=False)

    # 1.5 * IQR Bound Evaluation
    q1 = modelling_table["delivery_hours"].quantile(0.25)
    q3 = modelling_table["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_b = q1 - 1.5 * iqr
    upper_b = q3 + 1.5 * iqr
    outliers = ((modelling_table["delivery_hours"] < lower_b) | (modelling_table["delivery_hours"] > upper_b)).sum()

    # Store standard checks
    checks = {
        "raw_rows": raw_rows,
        "duplicate_rows_removed": dup_rows_removed,
        "rows_after_dedup": len(df_dedup),
        "missing_distance_km_before": missing_dist,
        "missing_parcel_weight_before": missing_weight,
        "sentinel_weight_rows": sentinels,
        "missing_declared_value_before": missing_val,
        "missing_payment_mode_before": missing_pay,
        "service_type_spellings_before": spellings_before,
        "service_type_spellings_after": spellings_after,
        "median_distance_km": round(medians["distance_km"], 4),
        "median_parcel_weight_kg": round(medians["parcel_weight_kg"], 4),
        "median_declared_value_inr": round(medians["declared_value_inr"], 4),
        "rows_after_hub_merge": len(modelling_table),
        "nulls_in_features_and_target": int(modelling_table[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum()),
        "delayed_count": int(modelling_table["is_delayed"].sum()),
        "delay_rate": round(float(modelling_table["is_delayed"].mean()), 4),
        "declared_value_mean_inr": round(modelling_table["declared_value_inr"].mean(), 4),
        "declared_value_median_inr": round(modelling_table["declared_value_inr"].median(), 4),
        "log_declared_value_mean": round(modelling_table["log_declared_value"].mean(), 4),
        "log_declared_value_median": round(modelling_table["log_declared_value"].median(), 4),
        "hours_q1": round(q1, 4),
        "hours_q3": round(q3, 4),
        "hours_lower_bound": round(lower_b, 4),
        "hours_upper_bound": round(upper_b, 4),
        "hours_outlier_count": outliers
    }

    pd.DataFrame(list(checks.items()), columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

    # Delay group evaluation
    records = []
    # Dimension 1: Service Type
    for g, group in modelling_table.groupby("service_type"):
        records.append(["service_type", g, len(group), group["is_delayed"].sum(), round(group["is_delayed"].mean(), 4), round(group["delivery_hours"].mean(), 4)])
    # Dimension 2: Hub Tier
    for g, group in modelling_table.groupby("hub_tier"):
        records.append(["hub_tier", str(int(g)), len(group), group["is_delayed"].sum(), round(group["is_delayed"].mean(), 4), round(group["delivery_hours"].mean(), 4)])
    # Dimension 3: Distance Bands
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_table["distance_band"] = pd.cut(modelling_table["distance_km"], bins=bins, labels=labels, right=True)
    for g, group in modelling_table.groupby("distance_band", observed=False):
        records.append(["distance_band", g, len(group), group["is_delayed"].sum(), round(group["is_delayed"].mean(), 4), round(group["delivery_hours"].mean(), 4)])

    pd.DataFrame(records, columns=["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]).to_csv("outputs/part_a_delay_by_group.csv", index=False)

    # Correlation logs
    num_cols = modelling_table.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = modelling_table[num_cols].corr().round(4)
    corr_matrix.insert(0, "feature", corr_matrix.index)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=False)

if __name__ == "__main__":
    run_part_a()
