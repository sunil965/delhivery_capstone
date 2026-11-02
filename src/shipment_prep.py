"""Shared data preparation and feature engineering module for Delhivery Delay Intelligence Pipeline.

This module provides reusable data cleaning, imputation, and feature engineering routines
shared between training (Part A) and scoring of new shipments (Part D / predict_delay).
"""

import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no",
]


def clean_shipments(raw: pd.DataFrame, medians: dict = None) -> tuple:
    """Cleans raw shipment records and imputes missing values using medians.

    Steps applied in order:
    i. drop exact duplicate rows;
    ii. service_type: strip leading and trailing spaces and convert to title case (Express, Standard, Economy);
    iii. payment_mode: replace missing values with placeholder 'Unknown';
    iv. declared_value: remove rupee sign and commas, convert to numeric declared_value_inr, drop text column;
    v. dispatch_time: convert to datetime;
    vi. parcel_weight_kg: values > 100 are scanner-fault sentinels -> treat as missing (NaN);
    vii. fill missing distance_km, parcel_weight_kg, declared_value_inr with column median.
         If medians is None, compute them from non-missing values remaining after steps i-vi.
         Otherwise, use the supplied medians dictionary.

    Returns:
        tuple: (clean_df, medians_dict)
    """
    df = raw.copy()

    # Step i: drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Step ii: service_type strip spaces and title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # Step iii: payment_mode fill missing with "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # Step iv: declared_value -> remove rupee sign and commas, convert to number declared_value_inr, drop declared_value
    def parse_currency(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).replace("₹", "").replace(",", "").strip()
        try:
            return float(val_str)
        except (ValueError, TypeError):
            return np.nan

    df["declared_value_inr"] = df["declared_value"].apply(parse_currency)
    df = df.drop(columns=["declared_value"])

    # Step v: dispatch_time convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # Step vi: parcel_weight_kg > 100 treat as missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # Step vii: fill missing distance_km, parcel_weight_kg, declared_value_inr
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attaches hub metadata and derives features required for regression and classification.

    Performs left merge on hub_id to add hub_city, region, hub_tier, hub_daily_capacity.
    Adds:
    - promised_hours (from PROMISED_HOURS, integer)
    - is_cod (1 when payment_mode is COD, else 0 - Unknown gives 0)
    - is_weekend (1 when dispatch_time day of week is Saturday or Sunday)
    - dispatch_hour (hour of dispatch timestamp as integer)
    - log_declared_value (natural log of declared_value_inr)

    Does not use delivery_hours, allowing scoring of undelivered shipments.
    """
    df = clean.copy()

    # Left merge hub columns on hub_id
    hub_cols = ["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]
    hubs_subset = hubs[hub_cols]
    df = df.merge(hubs_subset, on="hub_id", how="left")

    # promised_hours
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)

    # is_cod (1 for COD, 0 for Prepaid and Unknown)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # is_weekend (dayofweek >= 5: Saturday=5, Sunday=6)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # dispatch_hour
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # log_declared_value (natural log of declared_value_inr)
    df["log_declared_value"] = np.log(df["declared_value_inr"].astype(float))

    return df
