import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value",
    "n_stops", "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    """Cleans raw shipment dataframe following the strict sequential steps from Part A."""
    # drop exact duplicate rows
    df = raw.drop_duplicates().copy()

    # service_type: strip and convert to title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # payment_mode: replace missing values with 'Unknown'
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # declared_value: clean rupee symbol and commas, parse as float
    if "declared_value" in df.columns:
        df["declared_value_inr"] = (
            df["declared_value"]
            .astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["declared_value_inr"] = pd.to_numeric(df["declared_value_inr"], errors='coerce')
        df = df.drop(columns=["declared_value"])

    # dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # parcel_weight_kg: treat values > 100 as missing scanner-fault sentinels
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # fill missing values with medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median(skipna=True)),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median(skipna=True)),
            "declared_value_inr": float(df["declared_value_inr"].median(skipna=True))
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians

def build_features(clean, hubs):
    """
    Attaches hub metadata and derives model features safely.
    """
    # Left merge to attach hub columns without changing row counts
    df = clean.merge(hubs, on="hub_id", how="left")

    # Derive mandatory engineered columns
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = df["dispatch_time"].dt.dayofweek.isin([5, 6]).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df