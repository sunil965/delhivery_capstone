"""Part A — Data Foundation and Delay Evidence.

Cleans raw shipment records, derives features, builds the modelling table,
and computes delay checks, group comparisons, and correlations.
"""

import os
import numpy as np
import pandas as pd
from src.shipment_prep import FEATURES, NOISE_COLUMNS, clean_shipments, build_features


def run_part_a():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    raw_path = "data/raw_shipments.csv"
    hubs_path = "data/hubs.csv"

    if not os.path.exists(raw_path) or not os.path.exists(hubs_path):
        raise FileNotFoundError("Missing raw data files in data/. Run generate_data.py first.")

    raw_df = pd.read_csv(raw_path)
    hubs_df = pd.read_csv(hubs_path)

    # 1. Measure "Before" metrics on raw data after dedup (or raw counts)
    raw_rows = len(raw_df)
    dedup_raw = raw_df.drop_duplicates()
    duplicate_rows_removed = raw_rows - len(dedup_raw)
    rows_after_dedup = len(dedup_raw)

    missing_distance_km_before = int(dedup_raw["distance_km"].isna().sum())
    # missing_parcel_weight_before is the count of NaN before filling (sentinels are 999.0, not NaN)
    missing_parcel_weight_before = int(dedup_raw["parcel_weight_kg"].isna().sum())
    sentinel_weight_rows = int((dedup_raw["parcel_weight_kg"] > 100).sum())
    missing_declared_value_before = int(dedup_raw["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup_raw["payment_mode"].isna().sum())
    service_type_spellings_before = int(dedup_raw["service_type"].dropna().nunique())

    # 2. Clean shipments
    clean_df, medians = clean_shipments(raw_df)
    service_type_spellings_after = int(clean_df["service_type"].nunique())

    # 3. Build features and merge with hubs
    modelling_df = build_features(clean_df, hubs_df)
    rows_after_hub_merge = len(modelling_df)

    # 4. Add target is_delayed
    modelling_df["is_delayed"] = (
        modelling_df["delivery_hours"] > modelling_df["promised_hours"]
    ).astype(int)

    # 5. Write modelling_table.csv
    modelling_df.to_csv("data/modelling_table.csv", index=False)
    print(f"Modelling table saved with {len(modelling_df)} rows and {len(modelling_df.columns)} columns.")

    # 6. Nulls in features and target
    target_and_features = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_in_features_and_target = int(modelling_df[target_and_features].isna().sum().sum())

    delayed_count = int(modelling_df["is_delayed"].sum())
    delay_rate = round(float(modelling_df["is_delayed"].mean()), 4)

    # Hours percentiles and outlier bounds
    h_q1 = float(modelling_df["delivery_hours"].quantile(0.25))
    h_q3 = float(modelling_df["delivery_hours"].quantile(0.75))
    h_iqr = h_q3 - h_q1
    h_lower = h_q1 - 1.5 * h_iqr
    h_upper = h_q3 + 1.5 * h_iqr
    hours_outliers = int(
        ((modelling_df["delivery_hours"] < h_lower) | (modelling_df["delivery_hours"] > h_upper)).sum()
    )

    # Declared value statistics on median-filled column
    dv_mean = round(float(modelling_df["declared_value_inr"].mean()), 4)
    dv_median = round(float(modelling_df["declared_value_inr"].median()), 4)
    log_dv_mean = round(float(modelling_df["log_declared_value"].mean()), 4)
    log_dv_median = round(float(modelling_df["log_declared_value"].median()), 4)

    # Checks DataFrame
    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", round(medians["distance_km"], 4)),
        ("median_parcel_weight_kg", round(medians["parcel_weight_kg"], 4)),
        ("median_declared_value_inr", round(medians["declared_value_inr"], 4)),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", round(h_q1, 4)),
        ("hours_q3", round(h_q3, 4)),
        ("hours_lower_bound", round(h_lower, 4)),
        ("hours_upper_bound", round(h_upper, 4)),
        ("hours_outlier_count", hours_outliers),
        ("declared_value_mean_inr", dv_mean),
        ("declared_value_median_inr", dv_median),
        ("log_declared_value_mean", log_dv_mean),
        ("log_declared_value_median", log_dv_median),
    ]
    formatted_checks = []
    for m, v in checks:
        if isinstance(v, (int, np.integer)):
            formatted_checks.append({"metric": m, "value": int(v)})
        else:
            formatted_checks.append({"metric": m, "value": round(float(v), 4)})

    pd.DataFrame(formatted_checks).to_csv("outputs/part_a_checks.csv", index=False)
    print("outputs/part_a_checks.csv written.")

    # 7. Delay evidence table
    # distance band: 0-200, 201-400, 401-600, 601+
    modelling_df["distance_band"] = pd.cut(
        modelling_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    group_rows = []
    # Dimension 1: service_type
    for grp in ["Economy", "Express", "Standard"]:
        sub = modelling_df[modelling_df["service_type"] == grp]
        group_rows.append({
            "dimension": "service_type",
            "group": grp,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # Dimension 2: hub_tier
    for grp in [1, 2, 3]:
        sub = modelling_df[modelling_df["hub_tier"] == grp]
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(grp),
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # Dimension 3: distance_band
    for grp in ["0-200", "201-400", "401-600", "601+"]:
        sub = modelling_df[modelling_df["distance_band"] == grp]
        group_rows.append({
            "dimension": "distance_band",
            "group": grp,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    pd.DataFrame(group_rows).to_csv("outputs/part_a_delay_by_group.csv", index=False)
    print("outputs/part_a_delay_by_group.csv written.")

    # 8. Correlations
    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns
    corr_matrix = modelling_df[numeric_cols].corr().round(4)
    corr_out = corr_matrix.reset_index().rename(columns={"index": "feature"})
    corr_out.to_csv("outputs/part_a_correlations.csv", index=False)
    print("outputs/part_a_correlations.csv written.")


if __name__ == "__main__":
    run_part_a()
