"""
Feature engineering — strictly causal.

Rule applied to every feature (locked doc §3):
"Could this exist in a real payment-risk system, computed only from data
available up to the moment this transaction is authorized, before any
fraud label exists?" If no, it is excluded.

Baseline mean/std used for z-score features are calibrated ONLY on the
TRAIN period, then frozen and applied to val/test — mirroring how a real
system would calibrate on historical data and apply it going forward,
and avoiding any leakage of val/test distribution into the baseline.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/claude/fraud-spike-detector/data")
TRAIN_DAYS_END = 27  # exclusive — matches config.TRAIN_DAYS


def _rolling_by_group_keyed(df, group_col, value_col, window, agg, out_col):
    """
    Correctly-aligned groupby().rolling(): pandas' groupby().rolling().apply()
    returns results ordered by GROUP, not by original row order. Assigning
    `.values` back positionally (the naive approach) silently misaligns almost
    every row. This merges back on (group_col, timestamp) instead, which is
    the actual fix, not just a workaround.
    """
    tmp = df[[group_col, "timestamp", value_col]].set_index("timestamp")
    grouped = tmp.groupby(group_col)[value_col]
    if agg == "count":
        result = grouped.rolling(window, closed="both").count()
    elif agg == "mean":
        result = grouped.rolling(window, closed="both").mean()
    elif agg == "nunique":
        result = grouped.rolling(window, closed="both").apply(lambda x: len(set(x)), raw=True)
    else:
        raise ValueError(agg)

    result = result.reset_index().rename(columns={value_col: out_col})
    # merge back on the exact (group_col, timestamp) key — order-safe
    merged = df.merge(result, on=[group_col, "timestamp"], how="left")
    # if duplicate (group_col, timestamp) pairs exist, keep first match per original row
    if len(merged) != len(df):
        merged = merged.drop_duplicates(subset=["transaction_id"], keep="first").reset_index(drop=True)
    return merged[out_col].values


def add_rolling_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["is_declined"] = (df["status"] == "declined").astype(int)
    df["_card_code"] = df["card_id"].astype("category").cat.codes
    df["_ones"] = 1

    df["merchant_txn_count_5min"] = _rolling_by_group_keyed(df, "merchant_id", "_ones", "5min", "count", "_tmp")
    df["merchant_txn_count_10min"] = _rolling_by_group_keyed(df, "merchant_id", "_ones", "10min", "count", "_tmp")
    df["merchant_decline_rate_10min"] = _rolling_by_group_keyed(df, "merchant_id", "is_declined", "10min", "mean", "_tmp")
    df["ip_distinct_cards_10min"] = _rolling_by_group_keyed(df, "ip_subnet", "_card_code", "10min", "nunique", "_tmp")
    df["device_distinct_cards_10min"] = _rolling_by_group_keyed(df, "device_fingerprint", "_card_code", "10min", "nunique", "_tmp")

    # --- time since this card's previous transaction (large sentinel if first-ever) ---
    df["prev_txn_time_this_card"] = df.groupby("card_id")["timestamp"].shift(1)
    df["seconds_since_last_txn_card"] = (
        (df["timestamp"] - df["prev_txn_time_this_card"]).dt.total_seconds()
    )
    df["seconds_since_last_txn_card"] = df["seconds_since_last_txn_card"].fillna(999999)

    # --- calendar features (always available, not leakage) ---
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df = df.drop(columns=["_card_code", "_ones", "prev_txn_time_this_card"])
    return df


def add_baseline_zscore_features(df):
    """Baseline mean/std computed ONLY from train-period rows, then applied to all rows."""
    train_mask = df["day_index"] < TRAIN_DAYS_END

    # Merchant x hour-of-day baseline for txn velocity (5min count)
    vel_baseline = (
        df.loc[train_mask]
        .groupby(["merchant_id", "hour_of_day"])["merchant_txn_count_5min"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "vel_base_mean", "std": "vel_base_std"})
        .reset_index()
    )
    df = df.merge(vel_baseline, on=["merchant_id", "hour_of_day"], how="left")
    df["vel_base_std"] = df["vel_base_std"].replace(0, np.nan).fillna(df["vel_base_std"].median())
    df["velocity_zscore"] = (df["merchant_txn_count_5min"] - df["vel_base_mean"]) / df["vel_base_std"]

    # Merchant x hour-of-day baseline for the 10-min count too (used by the rule-based baseline detector)
    vel10_baseline = (
        df.loc[train_mask]
        .groupby(["merchant_id", "hour_of_day"])["merchant_txn_count_10min"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "vel10_base_mean", "std": "vel10_base_std"})
        .reset_index()
    )
    df = df.merge(vel10_baseline, on=["merchant_id", "hour_of_day"], how="left")
    df["vel10_base_std"] = df["vel10_base_std"].replace(0, np.nan).fillna(df["vel10_base_std"].median())
    df["velocity_zscore_10min"] = (df["merchant_txn_count_10min"] - df["vel10_base_mean"]) / df["vel10_base_std"]
    df = df.drop(columns=["vel10_base_mean", "vel10_base_std"])

    # Merchant-level amount baseline (train-only)
    amt_baseline = (
        df.loc[train_mask]
        .groupby("merchant_id")["amount"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "amt_base_mean", "std": "amt_base_std"})
        .reset_index()
    )
    df = df.merge(amt_baseline, on="merchant_id", how="left")
    df["amount_zscore"] = (df["amount"] - df["amt_base_mean"]) / df["amt_base_std"]

    df = df.drop(columns=["vel_base_mean", "vel_base_std", "amt_base_mean", "amt_base_std"])
    return df


FEATURE_COLUMNS = [
    "amount", "amount_zscore",
    "merchant_txn_count_5min", "merchant_txn_count_10min", "merchant_decline_rate_10min",
    "velocity_zscore", "velocity_zscore_10min",
    "ip_distinct_cards_10min", "device_distinct_cards_10min",
    "seconds_since_last_txn_card",
    "is_new_device", "is_new_geo",
    "hour_of_day", "day_of_week",
]

# Columns kept for audit/explanation/joining but NEVER passed to the model
METADATA_COLUMNS = [
    "transaction_id", "timestamp", "day_index", "merchant_id",
    "burst_id", "event_type", "label_fraud", "other_fraud",
]


def main():
    df = pd.read_parquet(DATA_DIR / "raw_transactions.parquet")
    print(f"Loaded {len(df)} rows")

    df = add_rolling_features(df)
    print("Rolling features computed")

    df = add_baseline_zscore_features(df)
    print("Baseline z-score features computed (calibrated on train period only)")

    keep_cols = METADATA_COLUMNS + FEATURE_COLUMNS
    out = df[keep_cols].copy()

    for split_name, day_range in [("train", (0, 27)), ("val", (27, 36)), ("test", (36, 45))]:
        split_df = out[(out.day_index >= day_range[0]) & (out.day_index < day_range[1])].reset_index(drop=True)
        split_df.to_parquet(DATA_DIR / f"{split_name}_features.parquet", index=False)
        print(f"  {split_name}_features.parquet: {len(split_df)} rows, fraud rate {split_df.label_fraud.mean():.4f}")

    return out


if __name__ == "__main__":
    main()
