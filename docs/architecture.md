# Fraud-Spike Detector — Technical Architecture & Scope
Razorpay Buildathon · Track 02: AI Risk Manager · Solo build, 3 days

---

## 1. Project Definition — What We Detect (and What We Don't)

**Critical fix to your framing first:** "Fraud-spike detector" taken literally (pure time-series anomaly detection on an aggregate signal) has no natural way to produce "precision/recall on a held-out test set" — that requires *labeled instances*, not just an anomaly score on a curve. If we build only an aggregate spike monitor, judges will ask "precision/recall of what, exactly?" and we won't have a clean answer.

**The fix:** define the "class of loss" as a specific fraud *pattern* — **burst/coordinated attacks** (card testing, BIN attacks, bot-driven checkout bursts) — and build:
- a **transaction-level supervised classifier** (gives us precision/recall/PR-AUC on labeled data), **plus**
- an **aggregate spike-detection layer** on top of it (gives us the "spike" story, time-to-detect, and burst-level metrics)

This satisfies both the track's evaluation bar *and* the "spike" framing honestly.

**We WILL detect:**
- Card testing attacks (many small-amount authorizations in a short window, often high decline rate)
- BIN/sequential-card enumeration attacks
- Bot-driven checkout bursts (sudden volume spike from new devices/geos in a short window)
- Coordinated promo/identity-abuse bursts (shared attributes across many transactions in a tight window)

**We will NOT attempt to detect:**
- One-off single-transaction card fraud unrelated to bursts (different problem, different features)
- Chargebacks / friendly fraud (post-transaction dispute behavior — that's a different track direction)
- Account-takeover *after* login (session/behavioral biometrics — out of scope)
- Money-laundering rings / multi-merchant collusion graphs (needs graph data we won't have time to model well)
- KYC/identity document fraud

Stating this boundary explicitly in the submission is itself a credibility signal — "honest scoping" is exactly what Track 02's bar rewards.

---

## 2. Product / Demo Concept

A **replay dashboard** that plays back a day of a merchant's transaction stream (with a speed slider), showing:
- Live transaction feed with per-transaction fraud score
- A rolling spike-detection chart (fraud-score rate over time) with flagged burst windows highlighted
- Click-through on any flagged transaction or burst window → explanation panel (top contributing features / shared attributes across the burst)
- A metrics tab: precision/recall/PR-AUC, cost curve, per-attack-type breakdown, time-to-detect — computed live from the held-out test set, not hardcoded
- An audit log tab: every alert, its explanation, threshold used, model version

This is a strong demo because it *shows the metrics being computed*, not just a static slide — judges can see the held-out evaluation happen.

---

## 3. System Architecture

```
[Synthetic Data Generator] → [Raw Transaction Log (schema-matched to Razorpay)]
              ↓
[Feature Engineering Pipeline] (strictly backward-looking, time-ordered)
              ↓
[Transaction-Level Model: LightGBM + Isolation Forest features] → fraud_score per txn
              ↓
[Spike Aggregation Layer: rolling window + EWMA/CUSUM over fraud_score] → burst alerts
              ↓
[Explainability Layer: SHAP (per-txn) + shared-attribute summary (per-burst)]
              ↓
[Audit Log: append-only JSON/SQLite] ←→ [Optional LLM narrative layer]
              ↓
[Backend: FastAPI serving inference + replay + metrics endpoints]
              ↓
[Dashboard: Streamlit — see critique in §10]
```

**Data flow for demo:** pre-trained model + pre-computed features on the *test* split → backend replays test-set transactions at a controllable speed → spike layer fires alerts in near-real-time → dashboard renders.

---

## 4. Dataset Design

**Schema (per transaction):**
```
transaction_id, timestamp, merchant_id, customer_id,
card_id, bin, device_fingerprint, ip_subnet,
amount, currency, payment_method, mcc,
geo_country, geo_city, is_new_device, is_new_geo,
status (authorized/declined/captured),
label_fraud (0/1), burst_id (null if benign — NEVER used as a feature)
```

**Volume:** ~200,000 transactions across **8–10 merchants**, spanning a simulated **45-day window**. Large enough for meaningful rolling statistics, small enough to generate and process comfortably in a day.

**Normal behavior:** diurnal + weekly seasonality per merchant, amount distributions conditioned on MCC, stable baseline velocity per merchant/customer.

**Injected fraud patterns (~15–25 distinct burst events):**
- Card-testing burst: 20–200 small transactions (₹1–₹50) in 2–10 minutes, from a narrow BIN range, high decline rate
- BIN enumeration: sequential-looking card numbers, rapid-fire, single IP/device cluster
- Bot checkout burst: sudden 5–10x volume spike from new devices + new geos within a short window
- Promo-abuse burst: many transactions sharing device fingerprint or email pattern in a tight window

Vary intensity/duration across bursts so the test set isn't "easier" than train.

**Class imbalance:** target 0.5–2% fraud rate overall — realistic, and forces us to justify PR-AUC/cost over accuracy.

**Split — time-based, not random:**
- Train: first 60% of the timeline
- Validation (threshold tuning): next 20%
- Held-out test: final 20%, containing burst *instances* not seen in training (same attack *types*, new specific occurrences)

**Leakage risks to explicitly guard against:**
- Rolling/window features must only look backward from time *t* — never use future rows to compute a "past 5-min velocity" feature.
- `burst_id` must never enter the feature set (that's the label in disguise).
- Random shuffling before splitting would leak future burst context into training — this is the single most common mistake in fraud-detection projects and judges will ask about it directly.

---

## 5. Model Strategy

**Your instinct toward Isolation Forest alone is a weak choice — here's why, and the fix.**

Isolation Forest alone is unsupervised: it can't be tuned against your labels, tends to have poor precision on structured tabular fraud data, and gives you no clean way to report "precision/recall on held-out test set" without post-hoc label matching that looks cherry-picked. Don't lead with it.

**Recommended: a small ensemble, each piece earning its place:**
1. **Baseline (for the report, not the demo):** a transparent rule/z-score model (e.g., transactions-per-minute z-score per merchant). This sets a floor and proves your ML model is actually adding value — judges respect a stated baseline.
2. **Primary model: LightGBM (or XGBoost)** — gradient-boosted trees on tabular features, handles class imbalance via `scale_pos_weight`, trains in seconds even on 200k rows, and gives you SHAP values for free. This is your main precision/recall driver.
3. **Secondary signal: Isolation Forest score as an *input feature*** to the LightGBM model (not a standalone detector) — helps catch novel burst shapes not well represented in training labels, without sacrificing supervised precision.
4. **Aggregation layer: EWMA or CUSUM** over the rolling mean of LightGBM fraud scores per merchant per time-bucket — this is what actually produces "spike" alerts, distinct from individual transaction flags.

This combination is defensible in a judge Q&A: each component has a clear, non-overlapping job.

---

## 6. Evaluation Methodology

- **Headline metric: PR-AUC** (not ROC-AUC — ROC is misleading under 1–2% positive rate).
- **Precision, recall, F1** at a chosen operating threshold, plus the full PR curve so judges see you didn't just pick the flattering point.
- **False-positive cost, made explicit and numeric:**
  - Define cost per false positive (e.g., customer friction / manual review cost, say ₹X per false alarm)
  - Define cost per false negative (average fraud amount lost per missed burst transaction)
  - Report **total expected cost** at several thresholds, and pick the threshold that minimizes it — not the F1-maximizing one. State this choice explicitly; it's exactly the "honest false-positive cost" the track bar asks for.
- **Burst-level metrics, not just row-level:**
  - Did we detect the burst *at all*? (burst-level recall)
  - **Time-to-detect**: how many transactions/minutes into a burst before the spike layer fires. This is a strong, judge-legible differentiator most submissions won't have.
- **Per-attack-type breakdown** (card-testing vs bot-burst vs promo-abuse performance separately) — a single blended number invites the "cherry-picked" accusation the track explicitly warns against.
- Report all of this on the **time-based held-out test set only**, with unseen burst instances — say so explicitly in the README.

---

## 7. Explainability Strategy

- **Per-transaction:** SHAP values → top 3 features surfaced in plain language, e.g. "velocity 12 tx/min vs merchant baseline 0.3/min," "new device + geo mismatch," "amount 3.2σ below merchant norm."
- **Per-burst:** aggregate summary of the flagged window — transaction count, mean fraud score, and shared attributes across the cluster (same BIN range, same IP subnet, same device fingerprint family). This is what turns "a pile of flagged rows" into "a spike story."
- **Audit trail:** append-only log — timestamp, model version, feature snapshot, threshold used, explanation text, outcome field (for future human review). This directly satisfies the buildathon-wide bar: "every money action explainable, bounded, and gated."

---

## 8. Where an LLM Genuinely Adds Value (and Where It Doesn't)

**Don't** use an LLM to score or classify transactions — a tuned LightGBM will always beat an LLM on structured tabular fraud detection, and using an LLM there would read as a gimmick to ML-literate judges.

**Do** use it for:
- Turning SHAP output + burst summary stats into a clear analyst-facing narrative ("This burst looks like card testing: 47 transactions, ₹1–₹30 each, over 4 minutes, from BIN range 4532xx, 89% declined.")
- Drafting a **bounded, human-approved** recommended action ("flag BIN range for step-up verification") — never auto-executing anything, staying strictly defense-only per the bar
- Natural-language Q&A over the structured audit log ("show me all bursts from BIN 4532xx last week") — this is real agentic value because it's grounded in your own structured data, not free-generated

---

## 9. Razorpay Integration — What's Actually Needed

You have zero Razorpay experience and 3 days — don't spend a full day on API auth/webhooks for a detection-focused track.

**Minimum viable integration:**
- Use Razorpay **Test Mode** with test API keys and Razorpay's published test card numbers to generate a small batch (50–100) of real test transactions.
- Use this only to **validate your synthetic schema** matches real Razorpay payment/order payload fields (payment_id, order_id, method, card network, status codes, etc.) and to shape your webhook-style event schema (`payment.captured`, `payment.failed`) correctly.
- State in the README that the bulk of the 200k-transaction dataset is synthetic *by design* (fraud bursts of the required size/diversity don't exist in a sandbox), but the schema is grounded in real Razorpay test-mode payloads and the system is designed to ingest live `payment.*` webhooks in production.

This gets you real-API credibility for a few hours of work instead of a day-plus.

---

## 10. Honest Critique of Scope Choices

- **React dashboard is the wrong call for a solo 3-day build.** It'll eat a full day you don't have, for polish that doesn't move the needle on this track's bar (which is about detection rigor, not UI). **Use Streamlit** — you get charts, tables, and interactivity in a few hundred lines, and it frees up nearly a full day for model/evaluation quality, which is what's actually being judged here.
- **Isolation Forest as your headline detector** — addressed in §5, demote to a feature input.
- **Pure aggregate-only "spike detector"** — addressed in §1, needs the labeled transaction layer underneath it.
- **Random train/test split** — addressed in §4, must be time-based.

---

## 11. 3-Day Implementation Plan

**Day 1 — Data**
- Hr 0–1: Repo scaffold, env setup
- Hr 1–2: Razorpay test-mode account, generate ~50–100 real test transactions, capture payload schema
- Hr 2–5: Build synthetic normal-transaction generator matching finalized schema
- Hr 5–8: Inject 15–25 burst events across attack types, finalize labeled dataset, EDA notebook
- **Deliverable:** finalized dataset + schema doc + EDA plots

**Day 2 — Model**
- Hr 0–3: Feature engineering pipeline (backward-looking only), time-based split
- Hr 3–6: Train baseline z-score, LightGBM, Isolation-Forest-as-feature; tune on validation
- Hr 6–8: Build EWMA/CUSUM spike layer; compute precision/recall/PR-AUC/cost/time-to-detect on held-out test
- **Deliverable:** trained model artifacts + metrics report (numbers locked in, not to be "improved" further under time pressure)

**Day 3 — Product & Submission**
- Hr 0–3: FastAPI backend (replay endpoint, inference, SHAP, audit log)
- Hr 3–5: Streamlit dashboard (replay view, spike chart, explanation drill-down, metrics tab)
- Hr 5–6: LLM narrative layer (burst summaries, Q&A over audit log)
- Hr 6–7: README, Project Objectives / Build Challenges write-up
- Hr 7–8: Record 5-min pitch video, final end-to-end test, push repo

---

## 12. MVP vs Stretch

**MVP (must ship):**
- Time-based dataset with clear burst injection
- Feature pipeline, LightGBM classifier with real precision/recall/PR-AUC/cost on held-out test
- Basic spike layer (rolling-window threshold count is an acceptable fallback if EWMA/CUSUM runs short on time)
- SHAP explainability at transaction level
- Audit log (even flat JSON is fine)
- Streamlit dashboard
- README with honestly-reported metrics table

**Stretch (cut first if behind schedule):**
- Isolation Forest ensemble feature
- Formal EWMA/CUSUM (vs simple rolling threshold)
- LLM narrative + Q&A layer
- Real Razorpay test-mode schema validation
- Time-to-detect metric
- Per-attack-type metric breakdown
- Cost-threshold optimization curve visualization

**Cut entirely if truly squeezed:** multi-merchant modeling (drop to 2–3 merchants), any React work, live streaming (batch replay with a speed slider is sufficient).

---

## 13. What Makes This Stand Out vs. a Basic Fraud Classifier

- The two-layer design (transaction classifier + aggregate spike detector) directly answers the "fraud-**spike**" framing instead of being a generic fraud model with a new name.
- Honest per-attack-type metrics and a cost-based threshold choice — most hackathon submissions report one flattering blended number; you won't.
- Time-to-detect as an operational metric — legible to non-ML judges and directly relevant to a risk team's real question ("how fast would this have caught it?").
- An audit trail that satisfies the buildathon-wide requirement ("explainable, bounded, gated") explicitly, not just this track's.
- Schema grounded in real Razorpay test-mode payloads even though the bulk dataset is synthetic — shows diligence without wasting a day on integration.

---

## Final Locked Stack — Do Not Revisit Mid-Build

| Layer | Choice |
|---|---|
| Data generation | Python (`pandas`, `numpy`, `faker`) |
| Real schema validation | Razorpay Test Mode API (~50–100 transactions, one-time) |
| Feature engineering | `pandas` rolling windows, strictly backward-looking |
| Primary model | LightGBM (binary classifier, `scale_pos_weight` for imbalance) |
| Secondary signal | Isolation Forest score → fed in as a LightGBM input feature |
| Spike aggregation | EWMA/CUSUM over rolling mean fraud score (fallback: rolling-window threshold count) |
| Explainability | SHAP (per-txn) + shared-attribute summarizer (per-burst) |
| Audit log | Append-only JSON or SQLite |
| Backend | FastAPI |
| Dashboard | Streamlit (not React) |
| LLM layer (stretch) | Anthropic API — narrative generation + Q&A over audit log only, never scoring |
| Evaluation | PR-AUC, precision/recall/F1 at cost-optimal threshold, per-attack-type breakdown, time-to-detect |

Lock this and start Day 1 on data generation — the biggest 3-day risk is spending too long on schema/model iteration on Day 2 and running out of time for the dashboard and video on Day 3.
