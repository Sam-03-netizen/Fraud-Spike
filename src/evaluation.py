"""
Shared evaluation logic — implements the FROZEN two-level detection rule
(locked doc §6). This module is used identically for the rule-based
baseline and for LightGBM so comparisons are apples-to-apples.

Definitions (frozen, do not change after committing):
- Transaction-level alert: fraud_score >= threshold_txn
- Burst-level alert: >=3 transaction-level alerts for the same merchant
  within any rolling 10-minute window
- A true burst is DETECTED if a burst-level alert's window overlaps the
  true burst window
- Time-to-detect = (alert fire time) - (true burst start time), for
  detected bursts only
- A burst-level alert with no overlapping true burst = false burst alert
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc as sk_auc, confusion_matrix

BURST_MIN_ALERTS = 3
BURST_WINDOW = "10min"


def transaction_level_metrics(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                precision=precision, recall=recall, f1=f1)


def pr_auc(y_true, scores):
    p, r, _ = precision_recall_curve(y_true, scores)
    return sk_auc(r, p)


def find_burst_windows(df_with_alerts, merchant_col="merchant_id", time_col="timestamp"):
    """Given a df with a boolean 'txn_alert' column, find burst-level alert
    windows: any rolling 10-min window (per merchant) containing >= BURST_MIN_ALERTS
    transaction-level alerts. Returns a list of (merchant_id, window_start, window_end)
    burst-alert events, collapsing overlapping firings into one alert per burst."""
    alerts = df_with_alerts[df_with_alerts["txn_alert"] == 1].sort_values(time_col)
    burst_alerts = []

    for merchant_id, grp in alerts.groupby(merchant_col):
        times = grp[time_col].values
        times = pd.to_datetime(times)
        n = len(times)
        i = 0
        active_start = None
        while i < n:
            # count alerts within BURST_WINDOW of times[i]
            window_end = times[i] + pd.Timedelta(BURST_WINDOW)
            j = i
            while j < n and times[j] <= window_end:
                j += 1
            count_in_window = j - i
            if count_in_window >= BURST_MIN_ALERTS:
                if active_start is None:
                    active_start = times[i]
                    fire_time = times[i + BURST_MIN_ALERTS - 1]  # fires on the 3rd alert
                    burst_alerts.append(dict(merchant_id=merchant_id, fire_time=fire_time,
                                              window_start=times[i], window_end=window_end))
                i += 1
            else:
                active_start = None
                i += 1
    return pd.DataFrame(burst_alerts)


def burst_level_metrics(burst_alerts_df, true_bursts_df):
    """true_bursts_df: one row per true burst with merchant_id, start_time, end_time.
    Returns detection rate (recall), false burst alert count, and time-to-detect list."""
    if burst_alerts_df.empty:
        return dict(detected=0, total_true_bursts=len(true_bursts_df), recall=0.0,
                    false_burst_alerts=0, time_to_detect_seconds=[])

    detected_burst_ids = set()
    time_to_detect = []
    matched_alert_idx = set()

    for _, burst in true_bursts_df.iterrows():
        candidates = burst_alerts_df[
            (burst_alerts_df.merchant_id == burst.merchant_id) &
            (burst_alerts_df.fire_time >= burst.start_time) &
            (burst_alerts_df.fire_time <= burst.end_time + pd.Timedelta(minutes=10))
        ]
        if len(candidates) > 0:
            detected_burst_ids.add(burst.burst_id)
            first_alert = candidates.iloc[0]
            ttd = (first_alert.fire_time - burst.start_time).total_seconds()
            time_to_detect.append(ttd)
            matched_alert_idx.add(candidates.index[0])

    total_alerts = len(burst_alerts_df)
    false_alerts = total_alerts - len(matched_alert_idx)

    return dict(
        detected=len(detected_burst_ids),
        total_true_bursts=len(true_bursts_df),
        recall=len(detected_burst_ids) / len(true_bursts_df) if len(true_bursts_df) > 0 else 0.0,
        false_burst_alerts=false_alerts,
        total_burst_alerts=total_alerts,
        burst_precision=len(matched_alert_idx) / total_alerts if total_alerts > 0 else 0.0,
        time_to_detect_seconds=time_to_detect,
    )


def get_true_bursts(raw_df):
    """Extract true burst windows (start/end/merchant) from the raw labeled data."""
    fraud = raw_df[raw_df.label_fraud == 1]
    bursts = fraud.groupby("burst_id").agg(
        merchant_id=("merchant_id", "first"),
        start_time=("timestamp", "min"),
        end_time=("timestamp", "max"),
    ).reset_index()
    return bursts
