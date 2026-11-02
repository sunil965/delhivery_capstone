import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def run_part_d():
    df = pd.read_csv("data/modelling_table.csv")
    hubs_meta = pd.read_csv("data/hubs.csv")

    # Task 1: Construct systematic profiles for each active location node
    hub_profile = df.groupby("hub_id").agg(
        shipments=("shipment_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean")
    ).reset_index()

    hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)
    hub_profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

    # Extract numerical matrices mapping locations sequentially
    cluster_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    X_hubs = hub_profile[cluster_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hubs)

    # Task 3: Evaluate Separation Curves across multiple configuration limits
    sep_records = []
    n_hubs = len(hub_profile)

    # Pre-calculate spatial distances mapped systematically
    all_pairs_dist = cdist(X_scaled, X_scaled, metric='euclidean')

    for k in:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        across_dists = []
        within_dists = []

        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = all_pairs_dist[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = np.mean(across_dists) / np.mean(within_dists)
        sizes = "-".join(map(str, sorted(pd.Series(labels).value_counts(), ascending=False)))
        sep_records.append([k, round(ratio, 4), sizes])

    pd.DataFrame(sep_records, columns=["K", "across_over_within", "cluster_sizes"]).to_csv("outputs/part_d_separation.csv", index=False)

    # Task 4: Finalize cluster tags (Enforcing business constraint K=3)
    km_3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster_label"] = km_3.fit_predict(X_scaled)

    # Map label definitions against absolute delays dynamically
    cluster_delay_means = hub_profile.groupby("cluster_label")["delay_rate"].mean().sort_values()
    rank_map = {cluster_label: name for rank, (cluster_label, _) in enumerate(cluster_delay_means.items()) for name in [["Standard", "Watch", "Escalate"][rank]]}

    hub_profile["segment"] = hub_profile["cluster_label"].map(rank_map)

    # Package configuration datasets matching downstream criteria
    hub_segments = hub_profile[["hub_id", "segment"]].merge(hubs_meta, on="hub_id", how="left")
    hub_segments[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/hub_segments.csv", index=False)

    seg_prof = hub_profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean")
    ).reindex(["Standard", "Watch", "Escalate"]).reset_index()
    seg_prof.round(4).to_csv("outputs/segment_profile.csv", index=False)

    # Evaluate structural operational alignment metrics
    tier_mapping = {1: "Standard", 2: "Watch", 3: "Escalate"}
    hub_segments["implied_segment"] = hub_segments["hub_tier"].map(tier_mapping)
    mismatches = hub_segments[hub_segments["segment"] != hub_segments["implied_segment"]]
    mismatches[["hub_id", "segment", "hub_city", "hub_tier"]].to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

if __name__ == "__main__":
    run_part_d()
