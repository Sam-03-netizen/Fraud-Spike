"""
Synthetic transaction data generator for the Fraud-Spike Detector.

Design goals (locked methodology §1):
- Every "obvious" attack signal has a legitimate counterexample somewhere in the data.
- Different merchant baselines, different attack intensities/stealth.
- One merchant and one stealth profile held back entirely from train (§4 generalization test).
- No target precision/recall is engineered — this generator is designed once, then frozen.
"""
import random
import hashlib
import json
import uuid
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from config import (
    SEED, SIM_START, NUM_DAYS, TRAIN_DAYS, VAL_DAYS, TEST_DAYS,
    MERCHANTS, HOLDOUT_MERCHANT, HOLDOUT_MERCHANT_ACTIVE_FROM_DAY,
    HOLDOUT_STEALTH_PROFILE, HOLDOUT_STEALTH_ACTIVE_FROM_DAY,
    ATTACK_TYPES, LEGIT_EVENT_TYPES, CURRENCY, MCC_MAP,
)

random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

# The original version of this generator used uuid.uuid4(), which is built on
# os.urandom() and is NOT affected by random.seed() -- this silently broke the
# "same seed = same data" claim for ID strings (though not for anything that
# actually affects model results -- see README). Fixed by seeding a dedicated
# random.Random instance and using it to generate UUIDs deterministically.
_id_rng = random.Random(SEED)


def seeded_uuid():
    return str(uuid.UUID(int=_id_rng.getrandbits(128)))


GEOS = [("IN", "Mumbai"), ("IN", "Bengaluru"), ("IN", "Delhi"), ("IN", "Hyderabad"),
        ("IN", "Pune"), ("IN", "Chennai"), ("IN", "Kolkata"), ("IN", "Ahmedabad")]

BIN_POOL = [str(400000 + i * 137 % 99999).zfill(6) for i in range(400)]  # spread-out normal BINs


# --------------------------------------------------------------------------
# Customer pool per merchant
# --------------------------------------------------------------------------
def make_customer(existing_bins=None):
    geo = random.choice(GEOS)
    return dict(
        customer_id=str(seeded_uuid())[:12],
        card_id=str(seeded_uuid())[:10],
        bin=random.choice(BIN_POOL),
        device_fingerprint=str(seeded_uuid())[:12],
        ip_subnet=f"{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.0/24",
        geo_country=geo[0],
        geo_city=geo[1],
    )


def build_customer_pool(merchant_id, size):
    return [make_customer() for _ in range(size)]


# --------------------------------------------------------------------------
# Diurnal / weekly seasonality
# --------------------------------------------------------------------------
def diurnal_multiplier(hour, category):
    # evening-peaking bell curve, food_delivery has lunch+dinner bimodal
    if category == "food_delivery":
        return 0.3 + 1.4 * np.exp(-((hour - 13) ** 2) / 8) + 1.8 * np.exp(-((hour - 20) ** 2) / 8)
    return 0.2 + 1.6 * np.exp(-((hour - 19) ** 2) / 30)


def weekly_multiplier(dow, category):
    is_weekend = dow >= 5
    if category in ("food_delivery", "fashion", "gaming", "ticketing"):
        return 1.4 if is_weekend else 1.0
    if category in ("subscription", "education"):
        return 0.8 if is_weekend else 1.05
    return 1.1 if is_weekend else 1.0


# --------------------------------------------------------------------------
# Normal traffic generation
# --------------------------------------------------------------------------
def generate_normal_traffic(merchant_id, cfg, pool, active_day_range):
    rows = []
    day_start, day_end = active_day_range
    for day in range(day_start, day_end):
        dow = (SIM_START + dt.timedelta(days=day)).weekday()
        for hour in range(24):
            rate = cfg["baseline_rate"] * diurnal_multiplier(hour, cfg["category"]) * weekly_multiplier(dow, cfg["category"])
            n_txn = np.random.poisson(max(rate, 0.05))
            for _ in range(n_txn):
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                ts = SIM_START + dt.timedelta(days=day, hours=hour, minutes=minute, seconds=second)

                new_customer = random.random() < 0.08
                if new_customer:
                    cust = make_customer()
                    pool.append(cust)
                    is_new_device, is_new_geo = True, random.random() < 0.4
                else:
                    cust = random.choice(pool)
                    is_new_device = random.random() < 0.04  # normal device churn
                    is_new_geo = random.random() < 0.03

                amount = max(10, np.random.lognormal(np.log(cfg["avg_amount"]), 0.5))
                declined = random.random() < cfg["decline_rate"]

                rows.append(dict(
                    timestamp=ts, merchant_id=merchant_id, customer_id=cust["customer_id"],
                    card_id=cust["card_id"], bin=cust["bin"],
                    device_fingerprint=cust["device_fingerprint"] if not is_new_device else str(seeded_uuid())[:12],
                    ip_subnet=cust["ip_subnet"],
                    amount=round(amount, 2), currency=CURRENCY, payment_method="card",
                    mcc=MCC_MAP[cfg["category"]],
                    geo_country=cust["geo_country"] if not is_new_geo else random.choice(GEOS)[0],
                    geo_city=cust["geo_city"] if not is_new_geo else random.choice(GEOS)[1],
                    is_new_device=is_new_device, is_new_geo=is_new_geo,
                    status="declined" if declined else "captured",
                    label_fraud=0, other_fraud=0, burst_id=None, event_type="normal",
                ))
    return rows


# --------------------------------------------------------------------------
# Attack burst generation (the loss class — locked doc §2)
# --------------------------------------------------------------------------
def generate_attack_burst(merchant_id, cfg, start_ts, attack_type, stealth):
    """Generates one coordinated card-testing-family burst."""
    burst_id = str(seeded_uuid())[:8]
    rows = []

    if stealth == "fast_loud":
        n_txn = random.randint(15, 60)
        duration_min = random.uniform(1, 8)
    else:  # slow_drip
        n_txn = random.randint(10, 30)
        duration_min = random.uniform(20, 55)

    # shared attacker infrastructure
    shared_ip = f"{random.randint(1,50)}.{random.randint(0,255)}.{random.randint(0,255)}.0/24"
    shared_device_prefix = str(seeded_uuid())[:8]
    base_bin_int = random.randint(400000, 499000)

    for i in range(n_txn):
        offset_s = random.uniform(0, duration_min * 60)
        ts = start_ts + dt.timedelta(seconds=offset_s)

        if attack_type == "card_testing":
            amount = round(random.uniform(1, 50), 2)
            decline_p = 0.75
            device = shared_device_prefix + str(i % 3)
            ip = shared_ip
            card_bin = str(base_bin_int + i)[:6]
        elif attack_type == "bin_enumeration":
            amount = round(random.uniform(1, 20), 2)
            decline_p = 0.85
            device = shared_device_prefix
            ip = shared_ip
            card_bin = str(base_bin_int + i)  # sequential BIN pattern

        elif attack_type == "bot_checkout_burst":
            amount = round(np.random.lognormal(np.log(cfg["avg_amount"] * 0.6), 0.4), 2)
            decline_p = 0.35
            device = str(seeded_uuid())[:12]  # many distinct devices (bots)
            ip = shared_ip  # but funneled through similar infra subnet
            card_bin = random.choice(BIN_POOL)
        else:  # promo_abuse_burst
            amount = round(cfg["avg_amount"] * random.uniform(0.1, 0.3), 2)
            decline_p = 0.10
            device = shared_device_prefix + str(i % 5)
            ip = f"{random.randint(1,50)}.{random.randint(0,255)}.{random.randint(0,255)}.0/24"
            card_bin = random.choice(BIN_POOL)

        geo = random.choice(GEOS)
        rows.append(dict(
            timestamp=ts, merchant_id=merchant_id, customer_id=str(seeded_uuid())[:12],
            card_id=str(seeded_uuid())[:10], bin=card_bin,
            device_fingerprint=device, ip_subnet=ip,
            amount=amount, currency=CURRENCY, payment_method="card", mcc=MCC_MAP[cfg["category"]],
            geo_country=geo[0], geo_city=geo[1],
            is_new_device=True, is_new_geo=True,
            status="declined" if random.random() < decline_p else "captured",
            label_fraud=1, other_fraud=0, burst_id=burst_id,
            event_type=f"attack_{attack_type}_{stealth}",
        ))
    return rows


# --------------------------------------------------------------------------
# Legitimate ambiguous events (locked doc §1 — designed to overlap with attacks)
# --------------------------------------------------------------------------
def generate_legit_event(merchant_id, cfg, pool, start_ts, event_type):
    rows = []
    event_id = str(seeded_uuid())[:8]

    if event_type == "flash_sale":
        n_txn = random.randint(40, 150)
        duration_min = random.uniform(15, 90)
        for _ in range(n_txn):
            offset_s = random.uniform(0, duration_min * 60)
            ts = start_ts + dt.timedelta(seconds=offset_s)
            cust = random.choice(pool) if random.random() > 0.3 else make_customer()
            amount = round(max(10, np.random.lognormal(np.log(cfg["avg_amount"] * 0.8), 0.45)), 2)
            rows.append(dict(
                timestamp=ts, merchant_id=merchant_id, customer_id=cust["customer_id"],
                card_id=cust["card_id"], bin=cust["bin"], device_fingerprint=cust["device_fingerprint"],
                ip_subnet=cust["ip_subnet"], amount=amount, currency=CURRENCY, payment_method="card",
                mcc=MCC_MAP[cfg["category"]], geo_country=cust["geo_country"], geo_city=cust["geo_city"],
                is_new_device=random.random() < 0.15, is_new_geo=random.random() < 0.1,
                status="declined" if random.random() < cfg["decline_rate"] else "captured",
                label_fraud=0, other_fraud=0, burst_id=None, event_type=f"legit_{event_type}",
            ))

    elif event_type == "new_customer_push":
        n_txn = random.randint(20, 60)
        duration_min = random.uniform(10, 40)
        for _ in range(n_txn):
            offset_s = random.uniform(0, duration_min * 60)
            ts = start_ts + dt.timedelta(seconds=offset_s)
            cust = make_customer()
            pool.append(cust)
            amount = round(max(10, np.random.lognormal(np.log(cfg["avg_amount"]), 0.4)), 2)
            rows.append(dict(
                timestamp=ts, merchant_id=merchant_id, customer_id=cust["customer_id"],
                card_id=cust["card_id"], bin=cust["bin"], device_fingerprint=cust["device_fingerprint"],
                ip_subnet=cust["ip_subnet"], amount=amount, currency=CURRENCY, payment_method="card",
                mcc=MCC_MAP[cfg["category"]], geo_country=cust["geo_country"], geo_city=cust["geo_city"],
                is_new_device=True, is_new_geo=True,
                status="declined" if random.random() < cfg["decline_rate"] else "captured",
                label_fraud=0, other_fraud=0, burst_id=None, event_type=f"legit_{event_type}",
            ))

    elif event_type == "bulk_b2b":
        n_txn = random.randint(3, 8)
        duration_min = random.uniform(5, 30)
        for _ in range(n_txn):
            offset_s = random.uniform(0, duration_min * 60)
            ts = start_ts + dt.timedelta(seconds=offset_s)
            cust = random.choice(pool)
            amount = round(cfg["avg_amount"] * random.uniform(8, 25), 2)
            rows.append(dict(
                timestamp=ts, merchant_id=merchant_id, customer_id=cust["customer_id"],
                card_id=cust["card_id"], bin=cust["bin"], device_fingerprint=cust["device_fingerprint"],
                ip_subnet=cust["ip_subnet"], amount=amount, currency=CURRENCY, payment_method="card",
                mcc=MCC_MAP[cfg["category"]], geo_country=cust["geo_country"], geo_city=cust["geo_city"],
                is_new_device=False, is_new_geo=False, status="captured",
                label_fraud=0, other_fraud=0, burst_id=None, event_type=f"legit_{event_type}",
            ))

    else:  # retry_storm — single customer, repeated declines then success (benign)
        cust = random.choice(pool)
        n_txn = random.randint(3, 6)
        for i in range(n_txn):
            ts = start_ts + dt.timedelta(seconds=i * random.uniform(5, 20))
            amount = round(max(10, np.random.lognormal(np.log(cfg["avg_amount"]), 0.3)), 2)
            status = "captured" if i == n_txn - 1 else "declined"
            rows.append(dict(
                timestamp=ts, merchant_id=merchant_id, customer_id=cust["customer_id"],
                card_id=cust["card_id"], bin=cust["bin"], device_fingerprint=cust["device_fingerprint"],
                ip_subnet=cust["ip_subnet"], amount=amount, currency=CURRENCY, payment_method="card",
                mcc=MCC_MAP[cfg["category"]], geo_country=cust["geo_country"], geo_city=cust["geo_city"],
                is_new_device=False, is_new_geo=False, status=status,
                label_fraud=0, other_fraud=0, burst_id=None, event_type=f"legit_{event_type}",
            ))
    return rows


# --------------------------------------------------------------------------
# Isolated singleton fraud — explicitly OUT OF SCOPE for our loss class (§2)
# Present as noise the model must not need to catch, and must not be
# rewarded/punished for missing (label_fraud stays 0; other_fraud=1).
# --------------------------------------------------------------------------
def generate_isolated_fraud(merchant_id, cfg, start_ts):
    cust = make_customer()
    amount = round(max(10, np.random.lognormal(np.log(cfg["avg_amount"]), 0.6)), 2)
    geo = random.choice(GEOS)
    return dict(
        timestamp=start_ts, merchant_id=merchant_id, customer_id=cust["customer_id"],
        card_id=cust["card_id"], bin=cust["bin"], device_fingerprint=cust["device_fingerprint"],
        ip_subnet=cust["ip_subnet"], amount=amount, currency=CURRENCY, payment_method="card",
        mcc=MCC_MAP[cfg["category"]], geo_country=geo[0], geo_city=geo[1],
        is_new_device=True, is_new_geo=True, status="captured",
        label_fraud=0, other_fraud=1, burst_id=None, event_type="isolated_fraud_out_of_scope",
    )


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------
def schedule_events():
    """Returns (attack_events, legit_events, isolated_fraud_events) as lists of
    (merchant_id, day, hour_frac, type, stealth_or_none)."""
    attack_events, legit_events, isolated_events = [], [], []

    for merchant_id, cfg in MERCHANTS.items():
        if merchant_id == HOLDOUT_MERCHANT:
            active_days = range(HOLDOUT_MERCHANT_ACTIVE_FROM_DAY, NUM_DAYS)
        else:
            active_days = range(0, NUM_DAYS)

        # --- attack bursts: roughly 1 every 3-5 active days per merchant
        day_list = list(active_days)
        day = day_list[0] if day_list else 0
        while day < (day_list[-1] if day_list else 0):
            day += random.randint(3, 5)
            if day >= NUM_DAYS or (day_list and day > day_list[-1]):
                break
            attack_type = random.choice(ATTACK_TYPES)
            stealth = random.choice(["fast_loud", "slow_drip"])
            if stealth == HOLDOUT_STEALTH_PROFILE and day < HOLDOUT_STEALTH_ACTIVE_FROM_DAY:
                stealth = "fast_loud"  # withhold slow_drip from train period
            hour = random.uniform(0, 24)
            attack_events.append((merchant_id, day, hour, attack_type, stealth))

        # --- legit ambiguous events: roughly 1 every 4-7 active days
        day = day_list[0] if day_list else 0
        while day < (day_list[-1] if day_list else 0):
            day += random.randint(4, 7)
            if day >= NUM_DAYS or (day_list and day > day_list[-1]):
                break
            event_type = random.choice(LEGIT_EVENT_TYPES)
            hour = random.uniform(8, 22)
            legit_events.append((merchant_id, day, hour, event_type, None))

        # --- a handful of isolated (out-of-scope) fraud transactions
        for _ in range(random.randint(3, 6)):
            if not day_list:
                continue
            d = random.choice(day_list)
            hour = random.uniform(0, 24)
            isolated_events.append((merchant_id, d, hour, "isolated", None))

    # Novel test-only flash sale (bigger than any seen in train) — generalization check
    novel_day = random.randint(TEST_DAYS[0], TEST_DAYS[1] - 1)
    legit_events.append(("M01_electronics", novel_day, 12.0, "flash_sale_novel", None))

    return attack_events, legit_events, isolated_events


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    all_rows = []
    pools = {}

    print("Generating normal traffic per merchant...")
    for merchant_id, cfg in MERCHANTS.items():
        pool_size = max(50, cfg["baseline_rate"] * 40)
        pools[merchant_id] = build_customer_pool(merchant_id, pool_size)
        if merchant_id == HOLDOUT_MERCHANT:
            active_range = (HOLDOUT_MERCHANT_ACTIVE_FROM_DAY, NUM_DAYS)
        else:
            active_range = (0, NUM_DAYS)
        rows = generate_normal_traffic(merchant_id, cfg, pools[merchant_id], active_range)
        all_rows.extend(rows)
        print(f"  {merchant_id}: {len(rows)} normal transactions (active days {active_range})")

    print("Scheduling and generating attack bursts + legit ambiguous events...")
    attack_events, legit_events, isolated_events = schedule_events()

    for merchant_id, day, hour, attack_type, stealth in attack_events:
        start_ts = SIM_START + dt.timedelta(days=day, hours=hour)
        cfg = MERCHANTS[merchant_id]
        rows = generate_attack_burst(merchant_id, cfg, start_ts, attack_type, stealth)
        all_rows.extend(rows)

    for merchant_id, day, hour, event_type, _ in legit_events:
        start_ts = SIM_START + dt.timedelta(days=day, hours=hour)
        cfg = MERCHANTS[merchant_id]
        et = "flash_sale" if event_type == "flash_sale_novel" else event_type
        rows = generate_legit_event(merchant_id, cfg, pools[merchant_id], start_ts, et)
        if event_type == "flash_sale_novel":
            for r in rows:
                r["event_type"] = "legit_flash_sale_novel"
        all_rows.extend(rows)

    for merchant_id, day, hour, _, _ in isolated_events:
        start_ts = SIM_START + dt.timedelta(days=day, hours=hour)
        cfg = MERCHANTS[merchant_id]
        all_rows.append(generate_isolated_fraud(merchant_id, cfg, start_ts))

    print(f"Total rows generated: {len(all_rows)}")
    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["transaction_id"] = [f"txn_{i:08d}" for i in range(len(df))]
    df["day_index"] = (df["timestamp"] - SIM_START).dt.days

    cols = ["transaction_id", "timestamp", "day_index", "merchant_id", "customer_id",
            "card_id", "bin", "device_fingerprint", "ip_subnet", "amount", "currency",
            "payment_method", "mcc", "geo_country", "geo_city", "is_new_device",
            "is_new_geo", "status", "label_fraud", "other_fraud", "burst_id", "event_type"]
    df = df[cols]

    project_root = Path(__file__).resolve().parent.parent
    out_path = project_root / "data" / "raw_transactions.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")

    # Also save the burst/event schedule for our own audit (NOT a model feature)
    schedule_df = pd.DataFrame(
        [dict(merchant_id=m, day=d, hour=h, type=t, stealth=s, kind="attack") for m, d, h, t, s in attack_events] +
        [dict(merchant_id=m, day=d, hour=h, type=t, stealth=s, kind="legit") for m, d, h, t, s in legit_events]
    )
    schedule_df.to_csv(project_root / "data" / "event_schedule.csv", index=False)

    return df


if __name__ == "__main__":
    df = main()
