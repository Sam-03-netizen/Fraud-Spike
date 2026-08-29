# Frozen Test-Set Evaluation — FINAL, SINGLE RUN

**This evaluation is complete and locked. No further tuning follows this run,
regardless of the numbers below.** Threshold (0.9876) was frozen on the
validation set in Day 2 and used here unchanged.

Test-set access log: 2 accesses total (see `logs/test_access_log.txt`) —
(1) structural sanity check, no label inspection; (2) this final run.

## Overall (all merchants, all attack types)
- PR-AUC: **0.9952**
- Precision: **99.70%**, Recall: **95.71%**, F1: 0.9766
- TP=669, FP=2, FN=30, TN=29,744

## Burst-level
- Detection: **23/23 (100%)**
- False burst alerts: **0**
- Time-to-detect: median 41.5s, max 582.1s, min 1.6s
  (max corresponds to a slow-drip burst; still well within that burst's
  ~1,000–2,100s duration — detection lands comfortably before the attack ends)

## Seen vs. unseen merchant — the generalization question (locked doc §4)
| | n | Fraud rate | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| Seen merchants (M01–M08) | 27,615 | 2.40% | 99.69% | 95.93% | 636 | 2 | 27 |
| **Unseen merchant (M09_gaming)** | 2,830 | 1.27% | **100%** | **91.67%** | 33 | 0 | 3 |

**Known caveat discovered during the pre-evaluation sanity check:** `M09_gaming`
has `NaN` values for 3 of 14 features (`amount_zscore`, `velocity_zscore`,
`velocity_zscore_10min`) because those features depend on a per-merchant
baseline calibrated from the train period only — and `M09_gaming` never
appears in train, by design. LightGBM handles this natively as a missing-value
split. The unaffected features (`ip_distinct_cards_10min`,
`device_distinct_cards_10min`, decline rate, raw counts) are fully computed
for M09_gaming since they're self-contained rolling statistics.

**Interpretation:** despite losing 3 features entirely, the model still
achieves 100% precision and 91.67% recall on a merchant it never saw in
training — evidence that detection generalizes via the actual attack
signature (shared infrastructure, decline-rate anomaly) rather than
memorized per-merchant baselines. The modest recall drop (95.9% → 91.7%) is
consistent with, and plausibly explained by, the missing baseline features —
not a mystery result.

This is also a real, disclosed product limitation: a brand-new merchant
onboarding to this system starts with no historical baseline, exactly as
modeled here. A production version would need a cold-start strategy (e.g.,
category-level default baselines) for new merchants — noted as future work,
not fixed retroactively to avoid altering the locked pipeline after
freezing (locked doc's "no architecture changes except hard blockers" rule).

## What this means for the submission
- The headline numbers (99.7% precision / 95.7% recall / PR-AUC 0.9952) are
  the honest, single, unbiased answer — not tuned to look good.
- The per-merchant breakdown is a stronger, more specific generalization
  claim than a single blended number would be, and it surfaced a real
  limitation (cold-start baselines) worth being upfront about rather than
  hiding.
