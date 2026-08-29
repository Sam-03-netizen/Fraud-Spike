"""
Explainability layer (locked doc §9). Everything here is generated
deterministically from SHAP values and aggregate statistics -- no LLM
involved. Templates render plain-language sentences from actual feature
values and their SHAP contributions.
"""
import numpy as np
import pandas as pd
import shap

FEATURE_COLUMNS = [
    "amount", "amount_zscore",
    "merchant_txn_count_5min", "merchant_txn_count_10min", "merchant_decline_rate_10min",
    "velocity_zscore", "velocity_zscore_10min",
    "ip_distinct_cards_10min", "device_distinct_cards_10min",
    "seconds_since_last_txn_card",
    "is_new_device", "is_new_geo",
    "hour_of_day", "day_of_week",
]

FEATURE_DISPLAY_NAMES = {
    "amount": "Transaction amount",
    "amount_zscore": "Amount vs. merchant norm",
    "merchant_txn_count_5min": "Merchant velocity (5-min)",
    "merchant_txn_count_10min": "Merchant velocity (10-min)",
    "merchant_decline_rate_10min": "Merchant decline rate (10-min)",
    "velocity_zscore": "Velocity z-score (5-min)",
    "velocity_zscore_10min": "Velocity z-score (10-min)",
    "ip_distinct_cards_10min": "Distinct cards on this IP (10-min)",
    "device_distinct_cards_10min": "Distinct cards on this device (10-min)",
    "seconds_since_last_txn_card": "Time since card's last transaction",
    "is_new_device": "New device",
    "is_new_geo": "New geography",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
}


def compute_baselines(train_df):
    """Median value of each feature among NORMAL (label_fraud=0) train transactions
    -- used purely for the plain-language 'typical value' context in explanations."""
    normal = train_df[train_df.label_fraud == 0]
    return {col: normal[col].median() for col in FEATURE_COLUMNS}


def render_feature_sentence(feature, value, baseline):
    if feature == "amount":
        return f"Transaction amount \u20b9{value:,.0f} (typical for this merchant: \u20b9{baseline:,.0f})"
    if feature == "amount_zscore":
        return f"Amount is {value:.1f} standard deviations from the merchant's typical amount"
    if feature in ("merchant_txn_count_5min", "merchant_txn_count_10min"):
        window = "5-min" if "5min" in feature else "10-min"
        return f"{value:.0f} transactions for this merchant in the last {window} (typical: {baseline:.1f})"
    if feature == "merchant_decline_rate_10min":
        return f"{value*100:.0f}% decline rate for this merchant in the last 10 min (typical: {baseline*100:.0f}%)"
    if feature in ("velocity_zscore", "velocity_zscore_10min"):
        window = "5-min" if feature == "velocity_zscore" else "10-min"
        return f"Transaction velocity is {value:.1f} std devs above normal for this merchant/hour ({window} window)"
    if feature == "ip_distinct_cards_10min":
        return f"{value:.0f} distinct cards seen on this IP subnet in the last 10 min (typical: {baseline:.1f})"
    if feature == "device_distinct_cards_10min":
        return f"{value:.0f} distinct cards seen on this device in the last 10 min (typical: {baseline:.1f})"
    if feature == "seconds_since_last_txn_card":
        if value >= 999999:
            return "First-ever transaction from this card"
        return f"This card last transacted {value:.0f} seconds ago"
    if feature == "is_new_device":
        return "New device not previously seen" if value else "Recognized device"
    if feature == "is_new_geo":
        return "New geographic location for this customer" if value else "Recognized location"
    if feature == "hour_of_day":
        return f"Occurred at {int(value):02d}:00"
    if feature == "day_of_week":
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return f"Occurred on {days[int(value) % 7]}"
    return f"{FEATURE_DISPLAY_NAMES.get(feature, feature)} = {value}"


class TransactionExplainer:
    def __init__(self, booster, train_df):
        self.explainer = shap.TreeExplainer(booster)
        self.baselines = compute_baselines(train_df)

    def explain_row(self, row, top_n=3):
        """row: a pandas Series with at least FEATURE_COLUMNS present."""
        X = row[FEATURE_COLUMNS].to_frame().T.astype(float)
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):  # binary classifier sometimes returns [class0, class1]
            shap_vals = shap_vals[1]
        shap_vals = np.asarray(shap_vals).flatten()

        order = np.argsort(-np.abs(shap_vals))[:top_n]
        explanations = []
        for i in order:
            feat = FEATURE_COLUMNS[i]
            val = row[feat]
            sentence = render_feature_sentence(feat, val, self.baselines[feat])
            direction = "+" if shap_vals[i] > 0 else "-"
            explanations.append(dict(feature=feat, value=val, shap_value=float(shap_vals[i]),
                                      direction=direction, sentence=sentence))
        return explanations

    def explain_burst(self, burst_txn_rows, top_n=2):
        """burst_txn_rows: dataframe of all transactions in a detected burst window."""
        n = len(burst_txn_rows)
        mean_score = burst_txn_rows["lgb_score"].mean() if "lgb_score" in burst_txn_rows else None

        shared_attrs = []
        for col, label in [("ip_subnet", "IP subnet"), ("device_fingerprint", "device fingerprint"),
                            ("bin", "card BIN")]:
            if col in burst_txn_rows.columns:
                top_val = burst_txn_rows[col].value_counts()
                if len(top_val) > 0 and top_val.iloc[0] / n >= 0.5:
                    pct = top_val.iloc[0] / n * 100
                    shared_attrs.append(f"{pct:.0f}% share {label} {top_val.index[0]}")

        X = burst_txn_rows[FEATURE_COLUMNS].astype(float)
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_shap = np.abs(np.asarray(shap_vals)).mean(axis=0)
        top_features = np.argsort(-mean_shap)[:top_n]
        top_feature_names = [FEATURE_DISPLAY_NAMES[FEATURE_COLUMNS[i]] for i in top_features]

        return dict(
            transaction_count=n,
            mean_fraud_score=float(mean_score) if mean_score is not None else None,
            shared_attributes=shared_attrs,
            top_driving_features=top_feature_names,
        )
