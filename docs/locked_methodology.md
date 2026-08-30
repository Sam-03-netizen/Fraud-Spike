# Fraud-Spike Detector — Locked Methodology & Final Architecture
Razorpay Buildathon · Track 02 · Solo, 3 days
**Status: FROZEN. Do not revise during the 72-hour build except for hard technical blockers.**

---

## 1. Dataset Integrity

**Design principle: every "obvious" signal must have a legitimate counterexample somewhere in the data, or the model is learning your generator, not fraud.**

Concretely include:

- **Legitimate traffic spikes:** flash-sale events, payday-adjacent volume increases, cricket/festival-driven demand — same order-of-magnitude velocity increase as some attack bursts, but no shared-infrastructure signal (diverse IPs/devices) and normal decline rate.
- **Legitimate new-device/new-customer surges:** a marketing push or referral campaign causing a cluster of new devices/geos in a short window — deliberately overlaps with the "bot burst" attack signature on device/geo features alone.
- **Legitimate unusual amounts:** occasional large B2B/bulk-buyer transactions, clearance-sale small-ticket clusters — overlaps with attack amount ranges from both directions.
- **Overlapping distributions by design:** at least 2–3 benign burst types should be statistically close to attack bursts on the *primary* features (velocity, new-device rate) but differ on secondary features (decline rate, infra-sharing, amount variance). This is what prevents the model from learning a trivial rate threshold.
- **Different merchant baselines:** 8–10 merchants across categories (electronics, food delivery, subscriptions, ticketing), each with its own velocity/amount profile — no global threshold should work.
- **Different fraud intensities/stealth levels:** include both "fast-loud" bursts (obvious, high volume, short) and "slow-drip" bursts (spread over 30–60 minutes at a rate close to legitimate variation) — the slow-drip cases are *intentionally hard* and some may not be caught. That's expected and should be reported, not hidden.
- **Noise/edge cases:** legitimate retry-after-decline (network blips), refunds, timezone artifacts, and — importantly — **isolated non-burst fraud** (a single stolen-card transaction with no coordination). This is explicitly **out of scope** for our loss class (see §2) and should be excluded from the loss-class label entirely, not folded in as noise the model is punished for missing.

**Do not tune the generator to hit a target precision/recall.** The generator is designed once, for realism and ambiguity, then frozen. Whatever numbers the frozen test set produces are the numbers we report.

**Freeze protocol:**
1. Generate the full 45-day dataset with a fixed random seed. Write it to disk once.
2. Compute a SHA-256 hash of the raw dataset file and the test-split file specifically. Commit both the data file (or a pointer/checksum if too large) and the hash to git **before writing any model training code.**
3. All threshold and hyperparameter tuning happens exclusively on the validation split.
4. The test set is touched by an evaluation script **at most twice**: once early to confirm the script runs without errors (no metric inspection), once for the final reported numbers. Every run is appended to a `metrics_log.txt` that is never overwritten — this is the actual enforcement mechanism for a solo dev with no one else to police you. Show this log in the submission.

---

## 2. The Single Loss Class — Coordinated Card-Testing Bursts

**Definition (this is also the ground-truth label rule for the generator and the alert rule in §6 — same definition, used twice):**

A window of activity for a single merchant qualifies as a **card-testing burst** if, within a short time span (defined as ≤10 minutes), **most** of the following hold:
- Multiple transactions (≥5) share an infrastructure signal: same IP subnet, same device fingerprint, or adjacent/sequential card BINs
- Transaction amounts are small or narrowly repeated, below the merchant's typical median
- Decline rate in the window is substantially elevated versus the merchant's rolling baseline
- Inter-transaction time is short and regular in a way inconsistent with a single normal customer
- The window has a distinguishable start and end (a spike), not gradual multi-day growth

**Explicitly NOT in scope:**
- A single isolated declined transaction
- Organic flash-sale volume growth without shared infrastructure or elevated declines
- Slow multi-day organic growth
- One-off large legitimate transactions
- Post-transaction fraud (chargebacks, friendly fraud), account takeover, KYC fraud, laundering rings

This boundary goes in the README verbatim. No expansion into generic fraud detection during the build.

---

## 3. Leakage Review — The Strict Rule

**Rule for every feature: "Could this exist in a real payment-risk system, computed only from data available up to the moment this transaction is authorized, before any fraud label exists?"** If no, it's excluded.

| Feature | Passes? | Why |
|---|---|---|
| Rolling txn velocity (per card/device/IP) | ✅ | Computable in real time from past events only |
| Amount, amount z-score vs. merchant/customer history | ✅ | Computable causally |
| Rolling decline rate in window | ✅ | Declines are known as they happen |
| new_device / new_geo flags | ✅ | Known at auth time |
| BIN, MCC | ✅ | Static metadata |
| Distinct-card count on this device/IP in rolling window | ✅ | Causal aggregation |
| `burst_id` | ❌ | Ground-truth artifact — encodes the label |
| Any "total burst size" or attack-type field | ❌ | Requires knowing the future of the same burst |
| Customer lifetime stats computed over the full dataset | ❌ unless recomputed causally | Must use only up-to-time-*t* history |
| Any generator-internal field (seed, synthetic ID prefix, exact-millisecond timestamp patterns) | ❌ | Not real, and often an unintentional artifact |

**Mandatory leakage smoke test:** train a throwaway model using only a feature that *should* be meaningless (e.g., transaction ID parity, millisecond timestamp digit). If it predicts fraud above chance, your generator has an unintended artifact — fix the generator before proceeding, don't just drop the feature and move on, since the artifact may be hiding elsewhere too.

---

## 4. Temporal Evaluation

- **Train:** days 1–27 (60%) — full seed, all merchants, all attack profiles *except* those explicitly held back (below).
- **Validation:** days 28–36 (20%) — threshold and hyperparameter tuning only.
- **Test:** days 37–45 (20%) — touched once, per the freeze protocol.

**To demonstrate generalization, not just memorization of your own generator:**
- Reserve at least one attack **stealth profile** (e.g., a slow-drip variant) to appear **only** in validation/test, never in train.
- Hold out **one full merchant** from training entirely; it appears only in test. Report metrics for "seen merchants" and "unseen merchant" **separately** — this is the single strongest generalization signal you can show a judge.
- Ensure test period contains at least one **novel legitimate spike** (a flash-sale shape not present in train) to test false-positive robustness on unseen benign patterns too.

---

## 5. Baseline and Proving LightGBM Helps

**Baseline (non-ML, fully transparent):** flag a merchant-time-window as a burst if the rolling 10-minute transaction count exceeds that merchant's (baseline mean + 3×std). No model, no tuning beyond that one constant.

**Proof of improvement:** run the *identical* evaluation protocol (same test set, same burst-alert definition from §6) on the baseline and on LightGBM+spike-layer. Compare at **matched operating points** — e.g., hold recall constant and compare precision, or hold false-positive budget constant and compare recall. Report as one side-by-side table.

**If LightGBM does not materially beat the baseline:** report that honestly. Don't reframe metrics to hide it. Instead, pivot the value proposition to what LightGBM adds beyond raw lift — per-transaction explainability, a tunable cost-based threshold, and burst-level clustering that a fixed threshold rule can't produce. A modest, honestly-reported lift with good explainability is more credible than an inflated one.

---

## 6. Two-Level Detection — Frozen Definitions

**Transaction-level alert:** fires when LightGBM's fraud probability ≥ `threshold_txn`, chosen on the validation set via cost-minimization (§7).

**Burst-level alert:** fires the moment **≥3 transaction-level alerts** occur for the same merchant within any rolling **10-minute** window. (Exact constants — 3 and 10 min — are tuned once on validation, then frozen before touching test.)

**Burst counted as DETECTED (for recall):** a burst-level alert's window overlaps the true injected burst window.

**Time-to-detect:** (burst-alert fire time) − (true burst start time), computed only for detected bursts.

**False burst alert:** a burst-level alert whose window does not overlap any true burst window — counts against burst-level precision.

This whole block must be committed to git **before** the test-evaluation script is run, so there's a timestamped record that it wasn't adjusted after seeing results.

---

## 7. False-Positive Cost — Honest, Not Invented

We do not know Razorpay's real cost structure, so we never present a single "true" number. Instead:

- Define **three explicit, labeled scenarios**, e.g.:
  - *Review-heavy:* cost per false positive ≈ ₹50 (analyst review time)
  - *Friction-heavy:* cost per false positive ≈ ₹200 (assumes blocking causes cart abandonment)
  - *Low-friction:* cost per false positive ≈ ₹20
  - Cost per false negative = the actual amount of the missed fraudulent transaction(s), computed from data, not assumed.
- For each scenario, plot **total expected cost vs. threshold** and mark the minimum. Show whether the optimal threshold shifts meaningfully across scenarios — report whichever answer is true.
- **Dashboard:** a "Cost & Trade-offs" tab with sliders for `cost_FP` and `cost_FN` a judge can move live, clearly labeled: *"We don't have access to Razorpay's actual cost structure — these are illustrative assumptions. Adjust to see how the optimal threshold shifts."*
- **README:** state the assumptions explicitly and note that real deployment would calibrate these with Razorpay's risk/finance teams.

---

## 8. Model Complexity — Is LightGBM Alone Enough?

Default answer: **yes, ship LightGBM alone.** Only add anything else if an experiment proves it's needed.

If time allows one ablation: train LightGBM with vs. without an Isolation Forest anomaly score as an extra input feature, compare validation PR-AUC. Only keep it if it produces a clear, reportable gain (not a rounding difference). If you don't have time to run the ablation, the default is to **not** include it and say so plainly: *"Not attempted due to time constraints; LightGBM alone met our evaluation targets."* Only invest more model complexity here if LightGBM alone is genuinely struggling — not as a default enhancement.

---

## 9. SHAP Explanations — Exact, Deterministic Content

**No LLM involved in either of these — template-rendered directly from SHAP output, so nothing is invented:**

**Per suspicious transaction:** the model's fraud probability, plus the top 3 SHAP-ranked features rendered via a fixed sentence template, e.g.:
- "rolling_5min_txn_count = 14 (merchant baseline ≈1.2) — largest contributor"
- "amount = ₹18 (merchant median ₹450) — small-amount pattern typical of card testing"
- "new_device = True"

**Per detected burst:** aggregate window stats — alerted transaction count, mean fraud score, and the dominant shared attributes across the cluster (e.g., "83% share IP subnet 103.21.x.x," "BIN range 453201–453210"), plus the average SHAP contribution across the window's transactions to explain what drove the cluster as a whole.

State explicitly in both the dashboard and README: *"Generated deterministically from SHAP values and aggregate statistics — no LLM involved."*

---

## 10. The 5-Minute Demo — One Story

Single merchant, single day, one continuous narrative:

1. **Calm baseline** — normal transaction rate ticking along on the replay dashboard.
2. **Legitimate spike** — a flash sale hits. **The system does not alert.** Say this out loud — it's your strongest false-positive-robustness beat.
3. **The attack begins** — a card-testing burst starts (small amounts, narrow BIN range, rising decline rate). Watch transaction-level alerts accumulate, then the burst-level alert fires. Show the **time-to-detect** counter explicitly (e.g., "detected 90 seconds / 6 transactions in").
4. **Explain it** — click into the burst, show the SHAP-driven explanation and shared-attribute summary.
5. **Prove it** — cut to the metrics tab: held-out test precision/recall/PR-AUC, explicitly labeled as computed on unseen future data.
6. **Own the uncertainty** — cut to the cost tab, move the sliders live, say plainly you don't know Razorpay's real costs.
7. **Close the loop** — one screen with the baseline-vs-LightGBM comparison table, proving the ML is earning its place.

Script the narration in advance (~800–900 words for 5 minutes) and rehearse the click sequence so there's no dead air.

---

## 11. Razorpay Relevance Without Live Data

- State explicitly and plainly in the README: **all data is synthetic; no real Razorpay or third-party payment data was used.**
- Ground relevance by aligning your schema and event model to Razorpay's **publicly documented** payment/webhook fields and event names (`payment.captured`, `payment.failed`, etc.) — cite the public docs, never imply private access.
- Frame the system, clearly labeled as a forward-looking architecture note, as "designed to consume Razorpay's real-time webhook stream in production" — not implemented or tested against live infrastructure.
- The problem framing itself (card-testing bursts as a fraud-risk category) is legitimately drawn from the buildathon's own brief.

---

## 12. The Skeptical Judge — 10 Hardest Questions

| # | Question | Evidence you must show |
|---|---|---|
| 1 | Did the model just learn your generator's artifacts, not real fraud signal? | Leakage audit table (§3), smoke-test result, ablation removing suspect features with no metric collapse |
| 2 | Did you tune against your test set? | Git-committed hash/timestamp of the frozen test set predating model code, `metrics_log.txt` showing ≤2 accesses |
| 3 | Does this generalize beyond merchants/attacks you designed for? | Held-out-merchant results and held-out-attack-profile results, reported separately |
| 4 | What does your false-positive rate mean for a real merchant? | Confusion matrix at chosen threshold, translated to "X false alerts per 10,000 transactions" |
| 5 | Why trust your cost numbers? | Explicit "illustrative assumptions" framing, 3-scenario sensitivity analysis |
| 6 | Does LightGBM actually beat a simple rule? | Matched-operating-point baseline comparison table |
| 7 | Is your detection fast enough to matter before a burst finishes? | Time-to-detect distribution, honest discussion of cases where detection is too slow |
| 8 | What if an attacker evades your velocity features (e.g., 1 txn/minute over days)? | Explicit "known limitations" section in README naming this evasion vector as future work |
| 9 | Are your SHAP explanations faithful or decorative? | SHAP is a standard, model-native method; no LLM paraphrasing; explanations align with domain intuition |
| 10 | How is this Razorpay-relevant if you never touched their systems? | Schema-alignment note against Razorpay's public webhook docs, explicit synthetic-data disclaimer |

---

## Final Locked Architecture

| Layer | Choice |
|---|---|
| Data generation | Python (`pandas`, `numpy`, `faker`), single fixed seed, frozen + hashed before modeling |
| Loss class | Coordinated card-testing bursts only (§2 definition, frozen) |
| Feature engineering | Causal, backward-looking only; every feature passes the §3 rule |
| Primary/only model | LightGBM (binary classifier, `scale_pos_weight` for imbalance) |
| Additional algorithms | None, unless a validation-set ablation proves material benefit (§8) |
| Spike aggregation | Simple rolling-window threshold rule (§6); EWMA/CUSUM only if substantial time remains |
| Explainability | SHAP (per-txn) + shared-attribute aggregation (per-burst); template-rendered, no LLM |
| Audit log | Append-only JSON or SQLite |
| Dashboard | Single Streamlit app calling detection code in-process — no FastAPI, no React |
| LLM | None in MVP; only added post-hoc if it proves genuinely valuable once the core system works |
| Razorpay integration | None live; schema aligned to Razorpay's public webhook docs, explicit synthetic-data disclaimer |
| Evaluation | Time-based split with held-out merchant + held-out attack profile; baseline comparison; 3-scenario cost sensitivity; burst-level + transaction-level metrics; single frozen final test run |

**This is locked.** Changes during the 72-hour build are permitted only for genuine technical blockers (e.g., a library doesn't work as expected) — not for scope creep or "just one more feature."
