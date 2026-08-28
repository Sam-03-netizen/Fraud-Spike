"""
Frozen configuration for synthetic data generation.
DO NOT modify after the dataset is generated and hashed — see logs/freeze_manifest.json
"""
import datetime

SEED = 42

SIM_START = datetime.datetime(2026, 1, 1, 0, 0, 0)
NUM_DAYS = 45

# Day-index boundaries (0-indexed, inclusive start / exclusive end)
TRAIN_DAYS = (0, 27)   # 27 days  (60%)
VAL_DAYS = (27, 36)    # 9 days   (20%)
TEST_DAYS = (36, 45)   # 9 days   (20%)

# --- Merchants -----------------------------------------------------------
# baseline_rate_per_hour: average normal transactions/hour at peak of day
# avg_amount / amount_std: normal transaction amount distribution (INR)
# decline_rate: baseline organic decline probability
MERCHANTS = {
    "M01_electronics":  dict(category="electronics",   baseline_rate=18, avg_amount=3500, amount_std=1800, decline_rate=0.04),
    "M02_food_delivery": dict(category="food_delivery", baseline_rate=40, avg_amount=350,  amount_std=180,  decline_rate=0.03),
    "M03_subscription": dict(category="subscription",  baseline_rate=10, avg_amount=499,  amount_std=250,  decline_rate=0.05),
    "M04_ticketing":    dict(category="ticketing",     baseline_rate=8,  avg_amount=1200, amount_std=900,  decline_rate=0.04),
    "M05_fashion":      dict(category="fashion",       baseline_rate=22, avg_amount=1800, amount_std=1200, decline_rate=0.035),
    "M06_grocery":      dict(category="grocery",       baseline_rate=30, avg_amount=650,  amount_std=300,  decline_rate=0.025),
    "M07_travel":       dict(category="travel",        baseline_rate=6,  avg_amount=8500, amount_std=5000, decline_rate=0.05),
    "M08_education":    dict(category="education",     baseline_rate=5,  avg_amount=2200, amount_std=1400, decline_rate=0.04),
    "M09_gaming":       dict(category="gaming",        baseline_rate=15, avg_amount=650,  amount_std=500,  decline_rate=0.045),
}

# Merchant reserved ENTIRELY for the test period — never appears in train/val.
# Used to measure generalization to an unseen merchant baseline (locked doc §4).
HOLDOUT_MERCHANT = "M09_gaming"
HOLDOUT_MERCHANT_ACTIVE_FROM_DAY = TEST_DAYS[0]

# Stealth profile reserved for val/test only — never appears in train.
# Used to measure generalization to an unseen attack shape (locked doc §4).
HOLDOUT_STEALTH_PROFILE = "slow_drip"
HOLDOUT_STEALTH_ACTIVE_FROM_DAY = VAL_DAYS[0]

# Attack types in scope (locked doc §2 — coordinated card-testing bursts only)
ATTACK_TYPES = ["card_testing", "bin_enumeration", "bot_checkout_burst", "promo_abuse_burst"]

# Legitimate ambiguous event types (ARE NOT fraud, designed to overlap with attack signatures)
LEGIT_EVENT_TYPES = ["flash_sale", "new_customer_push", "bulk_b2b", "retry_storm"]

CURRENCY = "INR"
MCC_MAP = {
    "electronics": "5732", "food_delivery": "5812", "subscription": "5968",
    "ticketing": "7922", "fashion": "5651", "grocery": "5411",
    "travel": "4722", "education": "8299", "gaming": "5816",
}
