"""
Day 2: baseline + LightGBM, evaluated on VALIDATION ONLY.
Test set is not touched here — see logs/test_access_log.txt for the only
permitted test accesses (locked doc §1 freeze protocol).
"""
import sys
sys.path.insert(0, "/home/claude/fraud-spike-detector/src")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from evaluation import (
    transaction_level_metrics, pr_auc, find_burst_windows,
    burst_level_metrics, get_true_bursts,
)

DATA_DIR = Path("/home/claude/fraud-spike-detector/data")

FEATURE_COLUMNS = [
    "amount", "amount_zscore",
    "merchant_txn_count_5min", "merchant_txn_count_10min", "merchant_decline_rate_10min",
    "velocity_zscore", "velocity_zscore_10min",
    "ip_distinct_cards_10min", "device_distinct_cards_10min",
    "seconds_since_last_txn_card",
    "is_new_device", "is_new_geo",
    "hour_of_day", "day_of_week",
]

train = pd.read_parquet(DATA_DIR / "train_features.parquet")
val = pd.read_parquet(DATA_DIR / "val_features.parquet")
raw = pd.read_parquet(DATA_DIR / "raw_transactions.parquet")
raw_val = raw[(raw.day_index >= 27) & (raw.day_index < 36)]

true_bursts_val = get_true_bursts(raw_val)
print(f"True bursts in validation period: {len(true_bursts_val)}")

# ---------------------------------------------------------------------
# BASELINE: rule-based, using velocity_zscore_10min directly as the score
# ---------------------------------------------------------------------
print("\n=== BASELINE (rolling velocity z-score, threshold=3.0) ===")
baseline_threshold = 3.0
val = val.copy()
val["baseline_score"] = val["velocity_zscore_10min"].fillna(0)
val["txn_alert"] = (val["baseline_score"] >= baseline_threshold).astype(int)

txn_metrics_base = transaction_level_metrics(val.label_fraud, val.baseline_score, baseline_threshold)
print("Transaction-level:", txn_metrics_base)

burst_alerts_base = find_burst_windows(val)  # val_features already has timestamp + merchant_id
burst_metrics_base = burst_level_metrics(burst_alerts_base, true_bursts_val)
print("Burst-level:", {k: v for k, v in burst_metrics_base.items() if k != "time_to_detect_seconds"})
if burst_metrics_base["time_to_detect_seconds"]:
    print(f"  median time-to-detect: {np.median(burst_metrics_base['time_to_detect_seconds']):.1f}s")

# ---------------------------------------------------------------------
# LIGHTGBM
# ---------------------------------------------------------------------
print("\n=== LIGHTGBM ===")
X_train, y_train = train[FEATURE_COLUMNS], train["label_fraud"]
X_val, y_val = val[FEATURE_COLUMNS], val["label_fraud"]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

model = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1,
)
model.fit(X_train, y_train)

val_scores = model.predict_proba(X_val)[:, 1]
val["lgb_score"] = val_scores

pr_auc_val = pr_auc(y_val, val_scores)
print(f"PR-AUC (validation): {pr_auc_val:.4f}")

# Threshold search on validation to pick operating point (this IS the validation set's job)
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_val, val_scores)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]
print(f"F1-optimal threshold on validation: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")

txn_metrics_lgb = transaction_level_metrics(y_val, val_scores, best_threshold)
print("Transaction-level @ F1-optimal threshold:", txn_metrics_lgb)

val["txn_alert"] = (val["lgb_score"] >= best_threshold).astype(int)
burst_alerts_lgb = find_burst_windows(val)
burst_metrics_lgb = burst_level_metrics(burst_alerts_lgb, true_bursts_val)
print("Burst-level:", {k: v for k, v in burst_metrics_lgb.items() if k != "time_to_detect_seconds"})
if burst_metrics_lgb["time_to_detect_seconds"]:
    print(f"  median time-to-detect: {np.median(burst_metrics_lgb['time_to_detect_seconds']):.1f}s")

# ---------------------------------------------------------------------
# Matched-operating-point comparison (locked doc §5)
# ---------------------------------------------------------------------
print("\n=== MATCHED-RECALL COMPARISON (baseline vs LightGBM) ===")
target_recall = txn_metrics_base["recall"]
# find LightGBM threshold that achieves ~same recall
idx = np.argmin(np.abs(recalls - target_recall))
matched_threshold = thresholds[min(idx, len(thresholds) - 1)]
matched_metrics = transaction_level_metrics(y_val, val_scores, matched_threshold)
print(f"Baseline recall: {target_recall:.4f}, precision: {txn_metrics_base['precision']:.4f}")
print(f"LightGBM @ matched recall {matched_metrics['recall']:.4f}, precision: {matched_metrics['precision']:.4f}")

model.booster_.save_model(str(DATA_DIR.parent / "artifacts" / "lgb_model_val_tuned.txt"))
print("\nModel saved to artifacts/lgb_model_val_tuned.txt")

# feature importance
imp = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
print("\nFeature importances:\n", imp)
