"""
Fraud-Spike Detector — Streamlit Dashboard

Single in-process app: no FastAPI, no separate backend. Streamlit calls the
detection/explanation code directly. Run with:
    streamlit run dashboard.py
(from the src/ directory, or `streamlit run src/dashboard.py` from project root)

IMPORTANT: this dashboard's replay view uses VALIDATION data only. The
frozen test set is never browsed transaction-by-transaction here -- only
aggregate metrics from the single frozen test run (once it exists) belong
on the Metrics tab. This keeps the "don't peek at test" discipline intact
even while iterating on the UI.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

from explain import TransactionExplainer, FEATURE_COLUMNS
from evaluation import find_burst_windows, burst_level_metrics, get_true_bursts, transaction_level_metrics, pr_auc
from cost_analysis import run_all_scenarios, COST_SCENARIOS
from audit_log import AuditLog

st.set_page_config(page_title="Fraud-Spike Detector", layout="wide")

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "lgb_model_val_tuned.txt"
THRESHOLD_TXN = 0.9876


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(ARTIFACT_PATH))


@st.cache_data
def load_data():
    train = pd.read_parquet(DATA_DIR / "train_features.parquet")
    val = pd.read_parquet(DATA_DIR / "val_features.parquet")
    raw = pd.read_parquet(DATA_DIR / "raw_transactions.parquet")
    raw_val = raw[(raw.day_index >= 27) & (raw.day_index < 36)].copy()
    return train, val, raw_val


@st.cache_resource
def load_explainer(_model, _train):
    return TransactionExplainer(_model, _train)


model = load_model()
train, val, raw_val = load_data()
val = val.merge(raw_val[["transaction_id", "ip_subnet", "device_fingerprint", "bin"]],
                 on="transaction_id", how="left")
val["lgb_score"] = model.predict(val[FEATURE_COLUMNS])
val["txn_alert"] = (val["lgb_score"] >= THRESHOLD_TXN).astype(int)
explainer = load_explainer(model, train)
true_bursts = get_true_bursts(raw_val)

st.title("🛡️ Fraud-Spike Detector")
st.caption("Razorpay Buildathon · Track 02: AI Risk Manager · Coordinated card-testing burst detection")

tab_replay, tab_metrics, tab_cost, tab_audit = st.tabs(
    ["📊 Replay & Explain", "📈 Metrics (Validation)", "💰 Cost & Trade-offs", "📋 Audit Log"]
)

# =====================================================================
# TAB 1: Replay & Explain
# =====================================================================
with tab_replay:
    st.markdown("**Note:** this replay uses the validation split only. The frozen test set "
                "is never browsed here — only aggregate metrics from it appear on the Metrics tab.")

    col1, col2 = st.columns(2)
    with col1:
        merchant = st.selectbox("Merchant", sorted(val.merchant_id.unique()))
    with col2:
        merchant_days = sorted(val[val.merchant_id == merchant].day_index.unique())
        day = st.selectbox("Day (validation period, day 27-35)", merchant_days)

    day_data = val[(val.merchant_id == merchant) & (val.day_index == day)].sort_values("timestamp")
    day_data = day_data.merge(raw_val[["transaction_id", "event_type"]], on="transaction_id", how="left",
                               suffixes=("", "_raw")) if "event_type" not in day_data.columns else day_data

    if len(day_data) == 0:
        st.warning("No transactions for this merchant/day.")
    else:
        # Minute-level timeline of transaction volume + alerts
        day_data["minute"] = day_data["timestamp"].dt.floor("1min")
        timeline = day_data.groupby("minute").agg(
            volume=("transaction_id", "count"),
            alerts=("txn_alert", "sum"),
        ).reset_index()
        st.line_chart(timeline.set_index("minute")[["volume", "alerts"]])

        # Burst-level alerts for this merchant/day
        burst_alerts = find_burst_windows(day_data)
        true_bursts_today = true_bursts[(true_bursts.merchant_id == merchant)]

        if len(burst_alerts) > 0:
            st.success(f"🚨 {len(burst_alerts)} burst-level alert(s) fired for this merchant/day")
            for _, alert in burst_alerts.iterrows():
                window_txns = day_data[(day_data.timestamp >= alert.window_start) &
                                        (day_data.timestamp <= alert.window_end)]
                with st.expander(f"Burst alert @ {alert.fire_time} ({len(window_txns)} transactions in window)"):
                    burst_exp = explainer.explain_burst(window_txns)
                    st.write(f"**{burst_exp['transaction_count']} transactions**, "
                             f"mean fraud score **{burst_exp['mean_fraud_score']:.3f}**")
                    if burst_exp["shared_attributes"]:
                        st.write("**Shared attributes:** " + "; ".join(burst_exp["shared_attributes"]))
                    st.write("**Top driving features:** " + ", ".join(burst_exp["top_driving_features"]))
        else:
            st.info("No burst-level alerts for this merchant/day — traffic looked normal, "
                    "even if there was a legitimate volume spike.")

        st.markdown("---")
        st.subheader("Transaction detail")
        alerted = day_data[day_data.txn_alert == 1]
        if len(alerted) > 0:
            txn_options = alerted.transaction_id.tolist()
            selected_txn = st.selectbox("Inspect an alerted transaction", txn_options)
            row = day_data[day_data.transaction_id == selected_txn].iloc[0]
            st.write(f"Fraud score: **{row.lgb_score:.4f}** | Amount: ₹{row.amount:,.0f} | "
                     f"Time: {row.timestamp}")
            for exp in explainer.explain_row(row):
                st.write(f"{exp['direction']} {exp['sentence']}  *(SHAP contribution: {exp['shap_value']:.2f})*")
        else:
            st.write("No transaction-level alerts for this merchant/day.")

# =====================================================================
# TAB 2: Metrics
# =====================================================================
with tab_metrics:
    st.success("**FINAL RESULTS** — frozen test set, single evaluation run (Day 5). "
               "See `logs/test_evaluation_report.md` for full detail and `logs/test_access_log.txt` "
               "for the access record.")

    test = pd.read_parquet(DATA_DIR / "test_features.parquet")
    test["lgb_score"] = model.predict(test[FEATURE_COLUMNS])
    raw_all = pd.read_parquet(DATA_DIR / "raw_transactions.parquet")
    raw_test_only = raw_all[raw_all.day_index >= 36]
    test = test.merge(raw_test_only[["transaction_id", "event_type"]], on="transaction_id",
                       how="left", suffixes=("", "_dup"))
    if "event_type_dup" in test.columns:
        test = test.drop(columns=["event_type_dup"])
    true_bursts_test = get_true_bursts(raw_test_only)

    txn_metrics_test = transaction_level_metrics(test.label_fraud, test.lgb_score, THRESHOLD_TXN)
    test["txn_alert"] = (test.lgb_score >= THRESHOLD_TXN).astype(int)
    burst_alerts_test = find_burst_windows(test)
    burst_metrics_test = burst_level_metrics(burst_alerts_test, true_bursts_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (test)", f"{txn_metrics_test['precision']:.1%}")
    c2.metric("Recall (test)", f"{txn_metrics_test['recall']:.1%}")
    c3.metric("PR-AUC (test)", f"{pr_auc(test.label_fraud, test.lgb_score):.4f}")
    c4.metric("Burst detection (test)", f"{burst_metrics_test['detected']}/{burst_metrics_test['total_true_bursts']}")

    st.subheader("Held-out merchant generalization (M09_gaming, never seen in train/val)")
    seen = test[test.merchant_id != "M09_gaming"]
    unseen = test[test.merchant_id == "M09_gaming"]
    seen_m = transaction_level_metrics(seen.label_fraud, seen.lgb_score, THRESHOLD_TXN)
    unseen_m = transaction_level_metrics(unseen.label_fraud, unseen.lgb_score, THRESHOLD_TXN)
    hc1, hc2 = st.columns(2)
    hc1.metric("Seen merchants — recall", f"{seen_m['recall']:.1%}")
    hc2.metric("Unseen merchant (M09_gaming) — recall", f"{unseen_m['recall']:.1%}",
               help="This merchant never appeared in train or validation. This is the strongest "
                    "generalization evidence in the project.")

    st.subheader("Per-attack-type recall (test)")
    breakdown = []
    for et in sorted(test[test.event_type.str.startswith("attack_", na=False)].event_type.unique()):
        sub = test[test.event_type == et]
        breakdown.append(dict(attack_type=et, n=len(sub), recall=(sub.lgb_score >= THRESHOLD_TXN).mean()))
    st.dataframe(pd.DataFrame(breakdown), width="stretch")
    st.caption("Note: `promo_abuse_burst_slow_drip` is the weakest case (~75% recall) — it's also the "
               "one attack/stealth combination absent from train and validation entirely.")

    with st.expander("Validation-set metrics (used for threshold tuning, shown for reference only)"):
        val_txn_metrics = transaction_level_metrics(val.label_fraud, val.lgb_score, THRESHOLD_TXN)
        val_burst_alerts = find_burst_windows(val)
        val_burst_metrics = burst_level_metrics(val_burst_alerts, true_bursts)
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Precision", f"{val_txn_metrics['precision']:.1%}")
        vc2.metric("Recall", f"{val_txn_metrics['recall']:.1%}")
        vc3.metric("PR-AUC", f"{pr_auc(val.label_fraud, val.lgb_score):.4f}")
        vc4.metric("Burst detection", f"{val_burst_metrics['detected']}/{val_burst_metrics['total_true_bursts']}")

    st.subheader("Time-to-detect (detected bursts only, test set)")
    if burst_metrics_test["time_to_detect_seconds"]:
        ttd = pd.Series(burst_metrics_test["time_to_detect_seconds"])
        st.write(f"Median: {ttd.median():.1f}s | Max: {ttd.max():.1f}s | Min: {ttd.min():.1f}s")
        st.bar_chart(ttd)

# =====================================================================
# TAB 3: Cost & Trade-offs
# =====================================================================
with tab_cost:
    st.markdown("We don't have access to Razorpay's actual cost structure. "
                "These are **illustrative assumptions** — adjust the slider to see how the optimal "
                "threshold shifts.")

    cost_fp = st.slider("Assumed cost per false positive (₹)", 5, 300, 50, step=5)
    thresholds = np.linspace(0.01, 0.99, 99)

    from cost_analysis import compute_cost_curve, find_optimal_threshold
    cost_df = compute_cost_curve(val.label_fraud, val.lgb_score, val.amount, cost_fp, thresholds)
    optimal = find_optimal_threshold(cost_df)

    st.line_chart(cost_df.set_index("threshold")["total_cost"])
    st.write(f"**Optimal threshold at this cost assumption: {optimal.threshold:.3f}** "
             f"→ {int(optimal.fp_count)} false positives, {int(optimal.fn_count)} false negatives "
             f"(₹{optimal.fn_amount:,.0f} missed fraud), total cost ₹{optimal.total_cost:,.0f}")

    st.subheader("Preset scenarios")
    results = run_all_scenarios(val.label_fraud, val.lgb_score, val.amount, thresholds)
    scenario_rows = []
    for name, r in results.items():
        opt = r["optimal"]
        scenario_rows.append(dict(scenario=name, cost_fp=f"₹{r['cost_fp']}",
                                   optimal_threshold=f"{opt.threshold:.3f}",
                                   false_positives=int(opt.fp_count), false_negatives=int(opt.fn_count),
                                   total_cost=f"₹{opt.total_cost:,.0f}"))
    st.dataframe(pd.DataFrame(scenario_rows), width="stretch")

# =====================================================================
# TAB 4: Audit Log
# =====================================================================
with tab_audit:
    st.write("Every alert is written to an append-only log with model version, threshold, "
             "and a deterministic SHAP-based explanation — no LLM involved.")

    if st.button("Regenerate audit log for currently selected merchant/day"):
        audit_path = PROJECT_ROOT / "logs" / "dashboard_session_audit_log.jsonl"
        audit_path.write_text("")  # fresh for this demo session
        audit = AuditLog(audit_path, ARTIFACT_PATH, THRESHOLD_TXN)
        day_alerts = val[(val.merchant_id == merchant) & (val.day_index == day) & (val.txn_alert == 1)]
        for _, row in day_alerts.iterrows():
            exps = explainer.explain_row(row)
            audit.log_transaction_alert(row.transaction_id, row.merchant_id, row.lgb_score, exps)
        st.success(f"Logged {len(day_alerts)} alert(s) to {audit_path.name}")

    audit_path = PROJECT_ROOT / "logs" / "dashboard_session_audit_log.jsonl"
    if audit_path.exists() and audit_path.stat().st_size > 0:
        audit = AuditLog(audit_path, ARTIFACT_PATH, THRESHOLD_TXN)
        records = audit.read_all()
        st.write(f"{len(records)} record(s) in current session log:")
        for rec in records[:20]:
            with st.expander(f"{rec['transaction_id']} — score {rec['fraud_score']:.3f}"):
                st.json(rec)
    else:
        st.info("No audit records yet — select a merchant/day with alerts and click the button above.")
