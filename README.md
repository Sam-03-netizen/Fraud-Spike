# Fraud-Spike Detector
**Razorpay Buildathon · Track 02: AI Risk Manager**

A two-level detection system for **coordinated card-testing bursts** — one
specific, narrowly-scoped class of fraud loss, detected with a supervised
transaction-level classifier and an aggregate burst-alerting layer on top,
evaluated on a genuinely held-out test set including a merchant and an
attack/stealth combination never seen in training.

---

## What this detects (and what it doesn't)

**In scope:** coordinated card-testing bursts — clusters of transactions within
a short window (≤10 min) sharing infrastructure signals (IP subnet, device
fingerprint, or adjacent card BINs), with small/narrow amounts and an elevated
decline rate. Covers four sub-patterns: card testing, BIN enumeration, bot
checkout bursts, and promo-abuse bursts — each in both "fast-loud" and
"slow-drip" stealth variants.

**Explicitly out of scope:** isolated single-transaction fraud, chargebacks,
friendly fraud, account takeover, KYC fraud, money-laundering rings. This
boundary is deliberate — see `docs/locked_methodology.md` §2.

---

## Headline results (frozen test set, single evaluation run)

| Metric | Value |
|---|---|
| Precision | **99.7%** |
| Recall | **95.7%** |
| PR-AUC | **0.9952** |
| Burst-level detection | **23/23 (100%)**, 0 false burst alerts |
| Median time-to-detect | **41.5s** |

**The strongest result in the project:** a merchant never seen during training
or validation (`M09_gaming`) achieves **100% precision, 91.7% recall** on the
frozen test set — real evidence of a transferable attack signature, not
merchant-specific memorization.

**Baseline comparison:** at matched recall (79.4%), the rule-based baseline
gets 51.0% precision; LightGBM gets **100%** — a genuine, test-confirmed
improvement, not a validation-set artifact.

**One honest weak point:** `promo_abuse_burst_slow_drip` scored 75% recall —
notably, this is also the one attack/stealth combination that, by chance,
never appeared in train or validation at all. Full breakdown, including this
result in context, is in `logs/test_evaluation_report.md`.

---

## How the numbers were produced (short version)

1. **Synthetic data, deliberately made hard to game.** 139,643 transactions
   across 9 merchants over 45 days, with legitimate flash sales, new-customer
   pushes, and bulk B2B orders specifically designed to overlap with attack
   signatures on the "obvious" features (see `docs/locked_methodology.md` §1).
2. **Time-based split**, not random — train (days 0–26), validation (27–35),
   test (36–44) — with one merchant and one attack-stealth profile withheld
   entirely from train/validation to test generalization.
3. **Every feature passes a causality check**: "could this exist in a real
   system before the label is known?" (§3). A leakage smoke test confirmed no
   generator artifacts leaked into the features.
4. **A real bug was found and fixed** (Day 2): a `groupby().rolling()`
   misalignment silently corrupted every rolling feature. Caught via
   per-attack-type breakdown, not a blended metric — full writeup in
   `logs/validation_log.md`.
5. **Two ablations were run, not assumed**: Isolation Forest as an added
   feature was tested and dropped (negligible gain); EWMA/CUSUM burst
   aggregation was tested and rejected in favor of the simpler rolling-
   threshold rule, which already performs at ceiling.
6. **The test set was touched exactly once** for final metrics — see
   `logs/test_access_log.txt` for the access record, and
   `logs/validation_log.md` for a documented anomaly in that log that was
   investigated and resolved via git history before the final run.

---

## Repository structure

```
fraud-spike-detector/
├── README.md                     — this file
├── docs/
│   ├── architecture.md           — original system design
│   └── locked_methodology.md     — frozen methodology (read this for full rigor)
├── src/
│   ├── config.py                 — frozen generator config (merchants, holdouts)
│   ├── generate_data.py          — synthetic transaction generator
│   ├── build_features.py         — causal feature engineering
│   ├── evaluation.py             — shared two-level detection + metrics
│   ├── train_and_evaluate_day2.py — baseline vs. LightGBM (validation)
│   ├── day3_isolation_forest_ablation.py
│   ├── day3_burst_aggregation_ablation.py
│   ├── explain.py                — SHAP-based explanations (no LLM)
│   ├── cost_analysis.py          — false-positive cost sensitivity
│   ├── audit_log.py              — append-only alert log
│   ├── dashboard.py              — Streamlit demo app
│   └── day5_final_test_evaluation.py — the single frozen test run
├── data/                         — raw + engineered train/val/test splits
├── artifacts/
│   └── lgb_model_val_tuned.txt   — trained model (validation-tuned threshold)
└── logs/
    ├── freeze_manifest.json      — SHA-256 hashes, dataset frozen before modeling
    ├── test_access_log.txt       — append-only record of every test-set access
    ├── validation_log.md         — full experiment history, bugs, and decisions
    └── test_evaluation_report.md — final frozen test results, in detail
```

## How to run

```bash
pip install pandas numpy faker pyarrow lightgbm shap scikit-learn streamlit
cd src
python generate_data.py                    # regenerates raw data
python build_features.py                   # rebuilds engineered features
python train_and_evaluate_day2.py          # baseline + LightGBM on validation
python day3_isolation_forest_ablation.py   # IF ablation (dropped)
python day3_burst_aggregation_ablation.py  # EWMA/CUSUM comparison (rejected)
python day5_final_test_evaluation.py       # the single frozen test run
streamlit run dashboard.py                 # interactive demo
```

**Determinism, precisely stated:** row counts, fraud rates, burst timing, every
engineered feature value, and every model performance metric (precision,
recall, PR-AUC, held-out-merchant results) reproduce identically across runs
and machines — verified independently on a Linux sandbox and a Windows laptop.
The one thing that does **not** reproduce byte-for-byte is the literal ID
strings (`customer_id`, `card_id`, `device_fingerprint`, `burst_id`), since
they're generated with `uuid.uuid4()`, which is not affected by `random.seed()`
— so `raw_transactions.parquet`'s exact hash will differ between runs even
though its statistical content, and every downstream result, does not. The
shipped data files in this repository are the exact, unmodified files the
results in this README were computed from — they are not meant to be
regenerated to reproduce the numbers, only to demonstrate that the *pipeline
logic* reproduces them independently.

## Razorpay relevance

All data is synthetic — no real Razorpay or third-party payment data was used.
The transaction schema is aligned to Razorpay's publicly documented payment/
webhook fields (`payment.captured`, `payment.failed`, etc.), and the system is
designed, as a forward-looking architecture note, to consume that event stream
in production. It has not been tested against live Razorpay infrastructure.

## Known limitations

- `promo_abuse_burst_slow_drip` recall (75%) is the weakest result, on the one
  attack/stealth combination absent from train/validation entirely.
- Cost-optimal thresholds shift somewhat between validation (0.02–0.42) and
  test (0.06–0.96) depending on scenario — a real deployment would recalibrate
  periodically rather than freezing a threshold indefinitely.
- Velocity-based features could plausibly be evaded by an attacker spreading
  transactions across days rather than minutes; this is out of scope for the
  current feature set and noted as future work.
- **Cold-start baseline gap for new merchants**: 3 features (`amount_zscore`,
  `velocity_zscore`, `velocity_zscore_10min`) depend on a per-merchant
  baseline computed from train-period data. A merchant with no train-period
  history (as `M09_gaming` has, by design) gets NaN on these features —
  LightGBM handles this via missing-value splits, and the held-out merchant
  still achieved 100% precision / 91.7% recall despite it, but a production
  deployment would need a fallback baseline (e.g., category-level defaults)
  for genuinely new merchants. Verified directly on the test set: confirmed
  via `test_features.parquet` inspection, not assumed.
- No LLM narrative layer was added — the core SHAP/template explanations were
  judged sufficient without it, per the "don't add complexity without proven
  need" principle applied throughout.
