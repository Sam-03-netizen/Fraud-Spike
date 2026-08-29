"""
Day 3, Experiment 2: does EWMA or CUSUM burst-aggregation beat the current
simple rolling-threshold rule (locked doc §6)? Only adopt if it materially
improves burst-level recall, false-alert rate, or time-to-detect.
All on VALIDATION. Test set not touched.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from evaluation import find_burst_windows, burst_level_metrics, get_true_bursts

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_COLUMNS = [
    "amount", "amount_zscore",
    "merchant_txn_count_5min", "merchant_txn_count_10min", "merchant_decline_rate_10min",
    "velocity_zscore", "velocity_zscore_10min",
    "ip_distinct_cards_10min", "device_distinct_cards_10min",
    "seconds_since_last_txn_card",
    "is_new_device", "is_new_geo",
    "hour_of_day", "day_of_week",
]

val = pd.read_parquet(DATA_DIR / "val_features.parquet")
raw = pd.read_parquet(DATA_DIR / "raw_transactions.parquet")
raw_val = raw[(raw.day_index >= 27) & (raw.day_index < 36)]
true_bursts_val = get_true_bursts(raw_val)

model = lgb.Booster(model_file=str(PROJECT_ROOT / "artifacts" / "lgb_model_val_tuned.txt"))
threshold_txn = 0.9876
val = val.copy()
val["lgb_score"] = model.predict(val[FEATURE_COLUMNS])
val["txn_alert"] = (val["lgb_score"] >= threshold_txn).astype(int)


# ---------------------------------------------------------------------
# METHOD 1 (current): simple rolling-threshold (>=3 alerts in 10-min window)
# ---------------------------------------------------------------------
burst_alerts_simple = find_burst_windows(val)
metrics_simple = burst_level_metrics(burst_alerts_simple, true_bursts_val)
print("=== METHOD 1: Simple rolling-threshold (current, locked §6) ===")
print({k: v for k, v in metrics_simple.items() if k != "time_to_detect_seconds"})
if metrics_simple["time_to_detect_seconds"]:
    print(f"  median time-to-detect: {np.median(metrics_simple['time_to_detect_seconds']):.1f}s")


# ---------------------------------------------------------------------
# METHOD 2: EWMA on per-minute alert counts, per merchant
# ---------------------------------------------------------------------
def build_minute_bins(df, merchant_col="merchant_id", time_col="timestamp"):
    df = df.copy()
    df["minute_bucket"] = df[time_col].dt.floor("1min")
    counts = df.groupby([merchant_col, "minute_bucket"])["txn_alert"].sum().reset_index()
    counts = counts.rename(columns={"txn_alert": "alert_count"})
    return counts


def ewma_burst_detection(val_df, true_bursts, alpha=0.5, k_sigma=4.0):
    counts = build_minute_bins(val_df)
    fired_events = []
    for merchant_id, grp in counts.groupby("merchant_id"):
        grp = grp.sort_values("minute_bucket").reset_index(drop=True)
        # baseline mean/std of alert_count per merchant (should be ~0 most of the time)
        mean_b, std_b = grp.alert_count.mean(), max(grp.alert_count.std(), 0.1)
        ewma = 0.0
        already_firing = False
        for _, row in grp.iterrows():
            ewma = alpha * row.alert_count + (1 - alpha) * ewma
            if ewma > mean_b + k_sigma * std_b:
                if not already_firing:
                    fired_events.append(dict(merchant_id=merchant_id, fire_time=row.minute_bucket))
                already_firing = True
            else:
                already_firing = False
    return pd.DataFrame(fired_events)


def cusum_burst_detection(val_df, true_bursts, k=0.5, h=5.0):
    counts = build_minute_bins(val_df)
    fired_events = []
    for merchant_id, grp in counts.groupby("merchant_id"):
        grp = grp.sort_values("minute_bucket").reset_index(drop=True)
        mean_b = grp.alert_count.mean()
        s = 0.0
        already_firing = False
        for _, row in grp.iterrows():
            s = max(0.0, s + (row.alert_count - mean_b - k))
            if s > h:
                if not already_firing:
                    fired_events.append(dict(merchant_id=merchant_id, fire_time=row.minute_bucket))
                already_firing = True
            else:
                already_firing = False
                s = 0.0
    return pd.DataFrame(fired_events)


def evaluate_burst_alerts(burst_alerts_df, true_bursts_df):
    """Same matching logic as evaluation.burst_level_metrics but for a
    (merchant_id, fire_time) dataframe without window_start/window_end."""
    if burst_alerts_df.empty:
        return dict(detected=0, total_true_bursts=len(true_bursts_df), recall=0.0,
                    false_burst_alerts=0, total_burst_alerts=0, time_to_detect_seconds=[])
    detected = set()
    ttd = []
    matched = set()
    for _, burst in true_bursts_df.iterrows():
        cands = burst_alerts_df[
            (burst_alerts_df.merchant_id == burst.merchant_id) &
            (burst_alerts_df.fire_time >= burst.start_time - pd.Timedelta(minutes=1)) &
            (burst_alerts_df.fire_time <= burst.end_time + pd.Timedelta(minutes=10))
        ]
        if len(cands) > 0:
            detected.add(burst.burst_id)
            first = cands.iloc[0]
            ttd.append((first.fire_time - burst.start_time).total_seconds())
            matched.add(cands.index[0])
    total = len(burst_alerts_df)
    return dict(
        detected=len(detected), total_true_bursts=len(true_bursts_df),
        recall=len(detected) / len(true_bursts_df) if len(true_bursts_df) else 0.0,
        total_burst_alerts=total, false_burst_alerts=total - len(matched),
        time_to_detect_seconds=ttd,
    )


print("\n=== METHOD 2: EWMA (alpha=0.5, k=4 sigma) ===")
ewma_alerts = ewma_burst_detection(val, true_bursts_val)
metrics_ewma = evaluate_burst_alerts(ewma_alerts, true_bursts_val)
print({k: v for k, v in metrics_ewma.items() if k != "time_to_detect_seconds"})
if metrics_ewma["time_to_detect_seconds"]:
    print(f"  median time-to-detect: {np.median(metrics_ewma['time_to_detect_seconds']):.1f}s")

print("\n=== METHOD 3: CUSUM (k=0.5, h=5) ===")
cusum_alerts = cusum_burst_detection(val, true_bursts_val)
metrics_cusum = evaluate_burst_alerts(cusum_alerts, true_bursts_val)
print({k: v for k, v in metrics_cusum.items() if k != "time_to_detect_seconds"})
if metrics_cusum["time_to_detect_seconds"]:
    print(f"  median time-to-detect: {np.median(metrics_cusum['time_to_detect_seconds']):.1f}s")

print("\n=== SUMMARY TABLE ===")
summary = pd.DataFrame([
    dict(method="Simple rolling-threshold (current)", recall=metrics_simple["recall"],
         false_alerts=metrics_simple["false_burst_alerts"],
         median_ttd=np.median(metrics_simple["time_to_detect_seconds"]) if metrics_simple["time_to_detect_seconds"] else None),
    dict(method="EWMA", recall=metrics_ewma["recall"], false_alerts=metrics_ewma["false_burst_alerts"],
         median_ttd=np.median(metrics_ewma["time_to_detect_seconds"]) if metrics_ewma["time_to_detect_seconds"] else None),
    dict(method="CUSUM", recall=metrics_cusum["recall"], false_alerts=metrics_cusum["false_burst_alerts"],
         median_ttd=np.median(metrics_cusum["time_to_detect_seconds"]) if metrics_cusum["time_to_detect_seconds"] else None),
])
print(summary.to_string(index=False))
