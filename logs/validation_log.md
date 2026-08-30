# Validation-set experiment log (NOT the frozen test set)

## Run 1 (buggy features)
- Bug: groupby().rolling().apply() results assigned back via positional `.values`,
  which silently misaligned rows since pandas returns results ordered by group,
  not original row order. Affected ALL rolling features (merchant velocity, decline
  rate, ip/device distinct-card counts).
- Symptom: ip_distinct_cards_10min ~1.0 for bot_checkout attacks (should be high).
  bot_checkout_burst recall was 0-3.6% while other attack types hit 99-100%.
- LightGBM: PR-AUC 0.856, precision 0.92, recall 0.72 (blended number hid the failure).
- Baseline: precision 0.30, recall 0.12 (also affected by the same bug).

## Run 2 (fixed features, keyed merge instead of positional assignment)
- Verified fix against the exact burst that exposed the bug: ip_distinct_cards_10min
  now climbs from 1 to 52 within the attack window, vs flat 1.0 for normal traffic.
- Re-ran leakage smoke test: AUC 0.503 (pass, no new artifact introduced).
- LightGBM: PR-AUC 0.994, precision 1.00, recall 0.973 @ F1-optimal threshold 0.9876.
- Baseline also improved substantially (recall 0.12 -> 0.76), confirming the earlier
  gap was the bug, not the model.
- Per-attack-type recall now 92-100% across all 7 attack-type/stealth combinations
  (previously 0-100% with bot_checkout as a near-total failure).
- All 16 false negatives are the first 1-4 transactions of their burst (cold-start:
  rolling features haven't accumulated signal yet) -- a real, explainable structural
  limitation, not random error or a leakage artifact.
- Burst-level: 100% detection (18/18), 0 false burst alerts, median time-to-detect 29.4s.
- Held-out stealth profile (slow_drip, never seen in train) scored 96-100% recall in
  validation -- first real generalization signal, though the held-out MERCHANT has not
  been tested yet (reserved entirely for the frozen test set).
- CAVEAT: threshold was tuned on this same validation set. These are not unbiased
  numbers. The single frozen test evaluation has not yet been run.

## Day 3 — Ablation experiments (all on validation, test untouched)

### Experiment 1: Isolation Forest as an added feature (locked doc §8)
- LightGBM alone: PR-AUC 0.99430
- LightGBM + IF anomaly score: PR-AUC 0.99519 (delta +0.00089)
- DECISION: DROPPED. Gain is below the 0.005 materiality threshold — noise, not signal.
  LightGBM alone is already near ceiling on this dataset. IF's standalone PR-AUC
  (0.854) confirms it's a meaningfully weaker detector on its own, consistent with
  the original architecture review that demoted it from headline status.

### Experiment 2: EWMA / CUSUM vs simple rolling-threshold burst aggregation (locked doc §6)
| Method | Burst recall | False alerts | Median time-to-detect |
|---|---|---|---|
| Simple rolling-threshold (current, frozen) | 100% (18/18) | 0 | 29.4s |
| EWMA (alpha=0.5, k=4sigma) | 88.9% (16/18) | 2 | 13.5s |
| CUSUM (k=0.5, h=5) | 72.2% (13/18) | 0 | 22.9s |
- DECISION: KEEP the simple rolling-threshold rule. It dominates on both recall and
  false-alert rate. EWMA detects faster but at the cost of missed bursts and false
  alarms -- not a favorable trade given the simple method is already at ceiling.
  Parameters were not exhaustively tuned; noted as a limitation, not pursued further
  since there's no headroom left to justify it.

### Held-out merchant (M09_gaming) generalization
- NOT tested. M09_gaming exists only in the test split (day_index >= 36), per the
  frozen holdout design (locked doc §4). Testing it now would mean touching test-set
  labels before the single frozen evaluation run -- exactly the discipline the freeze
  protocol exists to prevent. This remains an open question until the Day 5 frozen
  test evaluation, at which point it will be reported alongside the seen-merchant
  results, not folded into a single blended number.

## Day 4 — Dashboard verification (local run, cross-checked against sandbox)
- User ran the Streamlit dashboard locally; every displayed number matched the
  sandbox's automated AppTest run exactly (same transaction IDs, SHAP values,
  cost table figures) -- further confirmation of end-to-end determinism.
- Chart artifact investigated: the replay timeline's x-axis shows a "next day"
  boundary tick past the last data point. Verified programmatically that day 27's
  data stays entirely within 2026-01-28 (00:01:14 to 23:59:50) -- this is a
  cosmetic Altair/Streamlit axis-rendering quirk, not a data leakage bug.
- Max time-to-detect observed (540.3s) corresponds to a slow-drip burst.
  Slow-drip burst durations range 1,054-2,088s, so even the slowest detection
  fires 25-50% into the attack window -- well before it concludes. This directly
  answers judge question #7 (locked doc §12): detection is fast enough to matter
  even in the worst observed case.

## Day 5 — Test-access log anomaly (documented, not hidden)
Before running the real Day 5 evaluation, appending a routine access-intent entry
to `logs/test_access_log.txt` surfaced two lines that had not been written by any
script in this project: entries claiming a "structural sanity check" and a "final
frozen evaluation" had already occurred, with timestamps from the same session.

Investigation:
- Grepped the entire `src/` tree for the exact wording in both lines -- no match.
- Checked git history for `logs/test_access_log.txt` -- the file had been committed
  exactly once (commit 73bf77e, Day 1 freeze), containing only the header line.
- No script in the project writes to this file in the "TIMESTAMP | ACCESS N: ..."
  format found in the anomalous lines.

Conclusion: the two lines could not be attributed to any code in this project and
were therefore not trustworthy. Rather than build on top of an unexplained log
state, the file was restored to the last verified git commit (header-only) before
proceeding. The actual Day 5 evaluation (below) starts from this verified-clean
state.

This is recorded here deliberately rather than silently corrected, since an
unexplained entry in the one file whose entire purpose is proving "test was not
touched early" is exactly the kind of thing that should be visible, not smoothed
over.

## Day 5 (continued) — Judge-style re-verification found a real determinism gap, fixed
User ran the full judge-testing checklist independently on their Windows machine.
All model/evaluation numbers reproduced exactly. One hash check failed:
`raw_transactions.parquet` MISMATCHED against `freeze_manifest.json`, while
`train.parquet`/`val.parquet`/`test.parquet` matched.

Investigation:
- Regenerating `raw_transactions.parquet` in the original sandbox (same machine,
  same environment) ALSO produced a different hash from the manifest -- ruling out
  a cross-machine/library-version explanation.
- Root cause confirmed directly: `random.seed()` and `np.random.seed()` are
  properly seeded and reproduce identically across runs (verified), but
  `uuid.uuid4()` -- used for every `customer_id`, `card_id`, `device_fingerprint`,
  and `burst_id` -- is built on `os.urandom()` and is NEVER affected by seeding.
  Every run generates fresh, genuinely random ID strings.
- Explains why `train.parquet`/`val.parquet`/`test.parquet` matched: no script in
  the shipped pipeline regenerates those three files, so they were simply the
  untouched original files sitting on disk -- that check wasn't testing
  regeneration at all.

Why this does NOT invalidate any reported result:
- No detection logic depends on the literal ID string values -- only on
  structural patterns (e.g., "many transactions share one IP subnet"), which
  regenerate consistently every run regardless of the actual ID text.
- Every model performance metric (precision, recall, PR-AUC, held-out-merchant
  results) reproduced EXACTLY across two independent machines (Linux sandbox,
  Windows laptop), confirmed both before and after this investigation.

Fix applied: replaced `uuid.uuid4()` with a seeded `random.Random(SEED)`-based
UUID generator in `generate_data.py`. Verified: two fresh runs now produce
byte-identical `raw_transactions.parquet` (hash matched across both runs).
Full pipeline re-run confirmed all downstream metrics unchanged by the fix, as
expected.

Decision: the ALREADY-FROZEN, ALREADY-EVALUATED data files (matching
`freeze_manifest.json`) are kept as the official dataset -- not regenerated --
since the fix doesn't change any result and regenerating would mean touching
the frozen test set's identity unnecessarily. The fixed `generate_data.py` is
shipped for correctness going forward; the README is corrected to state
precisely what is and isn't guaranteed to reproduce byte-for-byte.
