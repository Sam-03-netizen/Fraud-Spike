# Frozen Test Set Evaluation — Final Results

**This evaluation was run exactly once.** Model and threshold were frozen on
validation (Day 2) and not touched here. See `logs/test_access_log.txt` for the
access record, and `logs/validation_log.md` for the full experiment history
including the Day 5 test-access-log anomaly that was investigated and resolved
before this run.

## Headline numbers

| Metric | Validation (Day 2) | **Test (frozen, final)** |
|---|---|---|
| Precision | 100.0% | **99.7%** |
| Recall | 97.3% | **95.7%** |
| PR-AUC | 0.9943 | **0.9952** |
| Burst-level detection | 18/18 (100%) | **23/23 (100%)** |
| False burst alerts | 0 | **0** |
| Median time-to-detect | 29.4s | **41.5s** |
| Max time-to-detect | 540.3s | **582.1s** |

Test performance is close to validation, not inflated or degraded — a reasonable
sign the validation-tuned threshold wasn't overfit to that specific split.

## The headline result: held-out merchant generalization

`M09_gaming` never appeared in train or validation at all — it exists only in
the test period (day 36+), by design (locked doc §4).

| | Seen merchants (n=27,615) | **Unseen merchant M09_gaming (n=2,830)** |
|---|---|---|
| Precision | 99.7% | **100.0%** |
| Recall | 95.9% | **91.7%** |
| False positives | 2 | **0** |

The model generalizes to a merchant it has never seen, with no drop in precision
and only a modest recall drop (95.9% → 91.7%). This is the single strongest piece
of evidence in the whole project that the model learned transferable attack
patterns rather than merchant-specific memorization.

**Root-cause note on the recall gap (verified programmatically, not assumed):**
`M09_gaming` has 100% NaN values in exactly 3 of 14 features —
`amount_zscore`, `velocity_zscore`, `velocity_zscore_10min` — confirmed by
direct inspection of `test_features.parquet`. The cause is in
`build_features.py`: per-merchant baseline mean/std for these z-score features
are computed only from train-period rows (`train_mask`), then merged in via a
left join on `merchant_id`. Since `M09_gaming` has zero rows in train by
design, it has no matching baseline row, so the merge produces NaN for all
its rows on these three features. LightGBM handles this natively via
missing-value splits, which is why the model still performs well — but this
is a genuine, previously undetected limitation, not a non-issue.

**Product implication:** a brand-new merchant onboarding to this system in
production would face the same cold-start gap — no historical baseline
exists yet. A real deployment would need a fallback (e.g., a category-level
default baseline) for new merchants. This is noted as future work rather
than fixed now, per the "no architecture changes except hard blockers" rule
locked before this evaluation — the model still meets its targets despite
the gap, so patching it retroactively would violate the same discipline that
makes this evaluation trustworthy in the first place.

## Per-attack-type recall (test)

| Attack type | n | Recall |
|---|---|---|
| card_testing_fast_loud | 253 | 99.6% |
| bin_enumeration_slow_drip | 82 | 97.6% |
| bot_checkout_burst_fast_loud | 99 | 97.0% |
| bot_checkout_burst_slow_drip | 82 | 93.9% |
| promo_abuse_burst_fast_loud | 135 | 94.8% |
| **promo_abuse_burst_slow_drip** | 48 | **75.0%** |

**Honest weak point:** `promo_abuse_burst_slow_drip` is both the lowest-recall
case AND a combination that never appeared in train or validation — the
generator happened to place it only in the test period. This is exactly the
kind of out-of-distribution case where degraded performance is expected, and
75% recall on a combination the model never trained on at all is a defensible,
explainable result rather than a hidden failure.

## Baseline vs. LightGBM (test, matched-recall comparison)

| | Baseline (rule-based) | LightGBM @ matched recall (79.4%) |
|---|---|---|
| Precision | 51.0% | **100.0%** |

Confirms the Day 2 validation finding holds on unseen test data: LightGBM is not
just fitting validation noise, it's a genuine, reproducible improvement.

## Cost-sensitivity (test, illustrative assumptions only)

| Scenario | cost_fp | Optimal threshold | Total cost |
|---|---|---|---|
| Review-heavy | ₹50 | 0.060 | ₹25,763 |
| Friction-heavy | ₹200 | 0.960 | ₹27,820 |
| Low-friction | ₹20 | 0.060 | ₹25,283 |

**Note:** optimal thresholds shift somewhat between validation and test
(e.g., Review-heavy: 0.020 on validation vs. 0.060 on test). This is expected —
these thresholds are tuned to the specific fraud/legitimate mix of whichever
split they're computed on, and the shift is a legitimate reason a real deployment
would recalibrate periodically rather than freeze a threshold forever. We report
it plainly rather than picking whichever number looks more stable.

## What this means for the submission

- The system meets every element of the track's bar: working detector, measured
  precision/recall on a genuinely held-out test set, honest false-positive cost
  framing, explainable/auditable output, defense-only design.
- The generalization result (held-out merchant) is the strongest evidence of
  real signal over memorization and should be the centerpiece of the pitch.
- The one honest weak point (promo_abuse_burst_slow_drip at 75% recall) is
  explainable and should be stated plainly, not hidden — it's exactly the kind
  of out-of-distribution case a judge would expect to see some degradation on.
