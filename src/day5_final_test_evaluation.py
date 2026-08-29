"""
Day 5: THE single frozen test evaluation.

Uses the exact model (artifacts/lgb_model_val_tuned.txt) and threshold
(0.9876) selected on validation in Day 2. NOTHING is retuned here.
This script is run exactly once for the numbers that go in the README.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb

from evaluation import (
    transaction_level_metrics, pr_auc, find_burst_windows,
    burst_level_metrics, get_true_bursts,
)
from cost_analysis import run_all_scenarios

DATA_DIR = PROJECT_ROOT / "data"
THRESHOLD_TXN = 0.9876  # FROZEN on validation, Day 2 -- not touched here

FEATURE_COLUMNS = [
    "amount", "amount_zscore",
    "merchant_txn_count_5min", "merchant_txn_count_10min", "merchant_decline_rate_10min",
    "velocity_zscore", "velocity_zscore_10min",
    "ip_distinct_cards_10min", "device_distinct_cards_10min",
    "seconds_since_last_txn_card",
    "is_new_device", "is_new_geo",
    "hour_of_day", "day_of_week",
]

print("=" * 70)
print("DAY 5 FROZEN TEST EVALUATION -- SINGLE RUN")
print("=" * 70)

test = pd.read_parquet(DATA_DIR / "test_features.parquet")
raw = pd.read_parquet(DATA_DIR / "raw_transactions.parquet")
raw_test = raw[raw.day_index >= 36].copy()
test = test.merge(raw_test[["transaction_id", "ip_subnet", "device_fingerprint", "bin"]],
                   on="transaction_id", how="left")

model = lgb.Booster(model_file=str(PROJECT_ROOT / "artifacts" / "lgb_model_val_tuned.txt"))
test["lgb_score"] = model.predict(test[FEATURE_COLUMNS])

true_bursts_test = get_true_bursts(raw_test)
print(f"\nTest set: {len(test)} transactions, {test.label_fraud.sum()} fraud "
      f"({test.label_fraud.mean()*100:.2f}%), {len(true_bursts_test)} true bursts")

# ---------------------------------------------------------------------
# 1. TRANSACTION-LEVEL METRICS
# ---------------------------------------------------------------------
print("\n--- 1. Transaction-level metrics (frozen threshold 0.9876) ---")
txn_metrics = transaction_level_metrics(test.label_fraud, test.lgb_score, THRESHOLD_TXN)
print(txn_metrics)
print(f"PR-AUC: {pr_auc(test.label_fraud, test.lgb_score):.4f}")

# ---------------------------------------------------------------------
# 2. BURST-LEVEL METRICS
# ---------------------------------------------------------------------
print("\n--- 2. Burst-level metrics (frozen rule: >=3 alerts in 10-min window) ---")
test["txn_alert"] = (test.lgb_score >= THRESHOLD_TXN).astype(int)
burst_alerts = find_burst_windows(test)
burst_metrics = burst_level_metrics(burst_alerts, true_bursts_test)
print({k: v for k, v in burst_metrics.items() if k != "time_to_detect_seconds"})
if burst_metrics["time_to_detect_seconds"]:
    ttd = burst_metrics["time_to_detect_seconds"]
    print(f"Time-to-detect: median={np.median(ttd):.1f}s, max={np.max(ttd):.1f}s, min={np.min(ttd):.1f}s")

# ---------------------------------------------------------------------
# 3. PER-ATTACK-TYPE BREAKDOWN
# ---------------------------------------------------------------------
print("\n--- 3. Per-attack-type recall ---")
attack_types = sorted(test[test.event_type.str.startswith("attack_", na=False)].event_type.unique())
for et in attack_types:
    sub = test[test.event_type == et]
    recall = (sub.lgb_score >= THRESHOLD_TXN).mean()
    print(f"  {et}: n={len(sub)}, recall={recall:.3f}")

# ---------------------------------------------------------------------
# 4. HELD-OUT MERCHANT GENERALIZATION (the big open question)
# ---------------------------------------------------------------------
print("\n--- 4. Held-out merchant generalization (M09_gaming, never seen in train/val) ---")
seen = test[test.merchant_id != "M09_gaming"]
unseen = test[test.merchant_id == "M09_gaming"]

print(f"SEEN merchants (n={len(seen)}, fraud={seen.label_fraud.sum()}):")
if seen.label_fraud.sum() > 0:
    print(" ", transaction_level_metrics(seen.label_fraud, seen.lgb_score, THRESHOLD_TXN))

print(f"\nUNSEEN merchant M09_gaming (n={len(unseen)}, fraud={unseen.label_fraud.sum()}):")
if unseen.label_fraud.sum() > 0:
    print(" ", transaction_level_metrics(unseen.label_fraud, unseen.lgb_score, THRESHOLD_TXN))
    unseen_attack_types = sorted(unseen[unseen.event_type.str.startswith("attack_", na=False)].event_type.unique())
    for et in unseen_attack_types:
        sub = unseen[unseen.event_type == et]
        recall = (sub.lgb_score >= THRESHOLD_TXN).mean()
        print(f"    {et}: n={len(sub)}, recall={recall:.3f}")
else:
    print("  WARNING: no fraud transactions for M09_gaming in test -- cannot evaluate.")

# ---------------------------------------------------------------------
# 5. BASELINE vs LIGHTGBM ON TEST (matched-recall comparison)
# ---------------------------------------------------------------------
print("\n--- 5. Baseline vs LightGBM on test (matched-recall comparison) ---")
baseline_threshold = 3.0
test["baseline_score"] = test["velocity_zscore_10min"].fillna(0)
baseline_metrics = transaction_level_metrics(test.label_fraud, test.baseline_score, baseline_threshold)
print(f"Baseline (rule-based): {baseline_metrics}")

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(test.label_fraud, test.lgb_score)
target_recall = baseline_metrics["recall"]
idx = np.argmin(np.abs(recalls[:-1] - target_recall))
matched_threshold = thresholds[idx]
matched_metrics = transaction_level_metrics(test.label_fraud, test.lgb_score, matched_threshold)
print(f"LightGBM @ matched recall {matched_metrics['recall']:.4f}: precision={matched_metrics['precision']:.4f}")

# ---------------------------------------------------------------------
# 6. COST-SENSITIVITY ON TEST
# ---------------------------------------------------------------------
print("\n--- 6. Cost-sensitivity scenarios (test set, frozen model score) ---")
results = run_all_scenarios(test.label_fraud, test.lgb_score, test.amount)
for name, r in results.items():
    opt = r["optimal"]
    print(f"  {name} (cost_fp=₹{r['cost_fp']}): optimal_threshold={opt.threshold:.3f}, "
          f"fp={int(opt.fp_count)}, fn={int(opt.fn_count)}, total_cost=₹{opt.total_cost:,.0f}")

print("\n" + "=" * 70)
print("END OF SINGLE FROZEN TEST EVALUATION")
print("=" * 70)
