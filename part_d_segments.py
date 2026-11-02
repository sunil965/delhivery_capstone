"""Part D — Hub Segments and Playbook.

Clusters the 24 logistics hubs using K-Means into operational segments
(Standard, Watch, Escalate), evaluates separation metrics across K=2..5,
and identifies hubs where operational reality mismatches designated tier.
"""

import os
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def run_part_d():
    os.makedirs("outputs", exist_ok=True)

    modelling_path = "data/modelling_table.csv"
    hubs_path = "data/hubs.csv"

    if not os.path.exists(modelling_path) or not os.path.exists(hubs_path):
        raise FileNotFoundError("modelling_table.csv or hubs.csv not found.")

    mod_df = pd.read_csv(modelling_path)
    hubs_df = pd.read_csv(hubs_path)

    # 1. Hub profile: group by hub_id and compute profile statistics
    profile_df = (
        mod_df.groupby("hub_id")
        .agg(
            shipments=("shipment_id", "count"),
            mean_delivery_hours=("delivery_hours", "mean"),
            delay_rate=("is_delayed", "mean"),
            mean_distance_km=("distance_km", "mean"),
            mean_n_stops=("n_stops", "mean"),
        )
        .reset_index()
        .sort_values("hub_id")
        .reset_index(drop=True)
    )

    # Round profile metrics to 4 decimal places
    for col in ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]:
        profile_df[col] = profile_df[col].round(4)

    profile_df.to_csv("outputs/hub_profile.csv", index=False)
    print("outputs/hub_profile.csv written.")

    # 2 & 3. K-Means segmentation & Separation evidence
    feature_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    # Profile is sorted by ascending hub_id
    X_scaled = scaler.fit_transform(profile_df[feature_cols])

    n_hubs = len(profile_df)
    pairs = list(combinations(range(n_hubs), 2))

    separation_records = []
    models = {}

    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        models[k] = (km, labels)

        across_dists = []
        within_dists = []

        for i, j in pairs:
            dist = np.linalg.norm(X_scaled[i] - X_scaled[j])
            if labels[i] == labels[j]:
                within_dists.append(dist)
            else:
                across_dists.append(dist)

        across_mean = np.mean(across_dists)
        within_mean = np.mean(within_dists)
        ratio = round(float(across_mean / within_mean), 4)

        counts = pd.Series(labels).value_counts().sort_values(ascending=False).tolist()
        cluster_sizes = "-".join(map(str, counts))

        separation_records.append({
            "K": k,
            "across_over_within": ratio,
            "cluster_sizes": cluster_sizes,
        })

    pd.DataFrame(separation_records).to_csv("outputs/part_d_separation.csv", index=False)
    print("outputs/part_d_separation.csv written.")

    # 4. Segment names and tier check for K = 3
    km_3, labels_3 = models[3]
    profile_df["cluster"] = labels_3

    # Order clusters by mean delay_rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_delay = (
        profile_df.groupby("cluster")["delay_rate"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    cluster_to_segment = {
        cluster_delay[0]: "Standard",
        cluster_delay[1]: "Watch",
        cluster_delay[2]: "Escalate",
    }
    profile_df["segment"] = profile_df["cluster"].map(cluster_to_segment)

    # outputs/hub_segments.csv (hub_id, segment, hub_city, hub_tier)
    hub_seg_df = profile_df[["hub_id", "segment"]].merge(
        hubs_df[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left"
    )
    hub_seg_df = hub_seg_df.sort_values("hub_id").reset_index(drop=True)
    hub_seg_df[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv(
        "outputs/hub_segments.csv", index=False
    )
    print("outputs/hub_segments.csv written.")

    # outputs/segment_profile.csv (segment, hubs, mean_delivery_hours, delay_rate, mean_distance_km, mean_n_stops)
    # Means of the hub-level values
    seg_profile_list = []
    for seg in ["Standard", "Watch", "Escalate"]:
        sub = profile_df[profile_df["segment"] == seg]
        seg_profile_list.append({
            "segment": seg,
            "hubs": len(sub),
            "mean_delivery_hours": round(float(sub["mean_delivery_hours"].mean()), 4),
            "delay_rate": round(float(sub["delay_rate"].mean()), 4),
            "mean_distance_km": round(float(sub["mean_distance_km"].mean()), 4),
            "mean_n_stops": round(float(sub["mean_n_stops"].mean()), 4),
        })
    pd.DataFrame(seg_profile_list).to_csv("outputs/segment_profile.csv", index=False)
    print("outputs/segment_profile.csv written.")

    # outputs/tier_vs_segment_mismatch.csv
    # Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_seg_df["expected_segment"] = hub_seg_df["hub_tier"].map(tier_expected_segment)
    mismatch_df = hub_seg_df[hub_seg_df["segment"] != hub_seg_df["expected_segment"]]
    mismatch_df[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv(
        "outputs/tier_vs_segment_mismatch.csv", index=False
    )
    print(f"outputs/tier_vs_segment_mismatch.csv written with {len(mismatch_df)} mismatches.")


if __name__ == "__main__":
    run_part_d()
