"""
Day 3, Experiment 1: Isolation Forest ablation (locked doc §8).

Rule: only keep IF as an added feature if it produces a MATERIAL gain in
validation PR-AUC over LightGBM alone. Default (null hypothesis) is to
drop it. Test set is not touched.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest

from evaluation import pr_auc, transaction_level_metrics

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

train = pd.read_parquet(DATA_DIR / "train_features.parquet")
val = pd.read_parquet(DATA_DIR / "val_features.parquet")

# --- Baseline: LightGBM alone (reproducing Day 2 result for a clean comparison) ---
scale_pos_weight = (train.label_fraud == 0).sum() / (train.label_fraud == 1).sum()
model_base = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1,
)
model_base.fit(train[FEATURE_COLUMNS], train.label_fraud)
val_scores_base = model_base.predict_proba(val[FEATURE_COLUMNS])[:, 1]
pr_auc_base = pr_auc(val.label_fraud, val_scores_base)
print(f"LightGBM alone -- validation PR-AUC: {pr_auc_base:.5f}")

# --- Candidate: + Isolation Forest anomaly score as an extra feature ---
# Fit IF on train features only (unsupervised, no label used), contamination
# roughly matching train fraud rate.
iso = IsolationForest(
    n_estimators=200, contamination=train.label_fraud.mean(), random_state=42
)
iso.fit(train[FEATURE_COLUMNS])

train_if = train.copy()
val_if = val.copy()
train_if["if_score"] = -iso.score_samples(train[FEATURE_COLUMNS])  # higher = more anomalous
val_if["if_score"] = -iso.score_samples(val[FEATURE_COLUMNS])

feature_cols_with_if = FEATURE_COLUMNS + ["if_score"]
model_if = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1,
)
model_if.fit(train_if[feature_cols_with_if], train_if.label_fraud)
val_scores_if = model_if.predict_proba(val_if[feature_cols_with_if])[:, 1]
pr_auc_if = pr_auc(val_if.label_fraud, val_scores_if)
print(f"LightGBM + IF score -- validation PR-AUC: {pr_auc_if:.5f}")

delta = pr_auc_if - pr_auc_base
print(f"\nDelta: {delta:+.5f}")

MATERIAL_THRESHOLD = 0.005  # a PR-AUC gain smaller than this is noise-level, not material
if delta > MATERIAL_THRESHOLD:
    print(f"DECISION: Isolation Forest feature KEPT (gain {delta:+.5f} > {MATERIAL_THRESHOLD} threshold)")
else:
    print(f"DECISION: Isolation Forest feature DROPPED (gain {delta:+.5f} <= {MATERIAL_THRESHOLD} threshold, "
          f"not material -- LightGBM alone is already near ceiling on this dataset)")

# Also check IF's OWN standalone anomaly-score PR-AUC, in case it's a genuinely
# strong signal that just doesn't add beyond what LightGBM already captures
pr_auc_if_alone = pr_auc(val.label_fraud, val_if["if_score"])
print(f"\n(For context) Isolation Forest score ALONE -- validation PR-AUC: {pr_auc_if_alone:.5f}")
