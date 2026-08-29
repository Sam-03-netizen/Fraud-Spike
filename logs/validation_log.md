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
