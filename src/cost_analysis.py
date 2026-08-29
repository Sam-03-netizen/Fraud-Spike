"""
False-positive cost analysis (locked doc §7). We do not know Razorpay's real
cost structure, so we never present one 'true' number -- three explicit,
labeled scenarios with a sensitivity sweep across thresholds instead.
"""
import numpy as np
import pandas as pd

COST_SCENARIOS = {
    "Review-heavy": dict(cost_fp=50, description="Cost per false positive = analyst manual review time"),
    "Friction-heavy": dict(cost_fp=200, description="Cost per false positive = assumed cart-abandonment/customer friction"),
    "Low-friction": dict(cost_fp=20, description="Cost per false positive = minimal automated soft-block cost"),
}


def compute_cost_curve(y_true, scores, amounts, cost_fp, thresholds=None):
    """Total cost = FP_count * cost_fp + sum(amount of missed fraud transactions).
    Returns a dataframe of threshold -> total_cost, fp_count, fn_amount."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts)

    for t in thresholds:
        preds = (scores >= t).astype(int)
        fp_mask = (preds == 1) & (y_true == 0)
        fn_mask = (preds == 0) & (y_true == 1)
        fp_count = fp_mask.sum()
        fn_amount = amounts[fn_mask].sum()
        total_cost = fp_count * cost_fp + fn_amount
        rows.append(dict(threshold=t, fp_count=int(fp_count), fn_count=int(fn_mask.sum()),
                          fn_amount=float(fn_amount), total_cost=float(total_cost)))
    return pd.DataFrame(rows)


def find_optimal_threshold(cost_df):
    idx = cost_df.total_cost.idxmin()
    return cost_df.loc[idx]


def run_all_scenarios(y_true, scores, amounts, thresholds=None):
    """Returns {scenario_name: (cost_df, optimal_row)} for all three scenarios."""
    results = {}
    for name, cfg in COST_SCENARIOS.items():
        cost_df = compute_cost_curve(y_true, scores, amounts, cfg["cost_fp"], thresholds)
        optimal = find_optimal_threshold(cost_df)
        results[name] = dict(cost_df=cost_df, optimal=optimal, cost_fp=cfg["cost_fp"],
                              description=cfg["description"])
    return results
