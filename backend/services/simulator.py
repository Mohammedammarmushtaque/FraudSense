"""
backend/services/simulator.py
FraudSense — Diverse Transaction Simulator

Generates realistic, varied transactions with controlled distribution:
  40% SAFE  — low amount, known device, home location
  35% MFA   — medium amount, foreign location or semi-risky merchant
  25% BLOCK — high amount, new device, risky merchant

Randomizes: location, device, merchant category, amounts, timestamps.
"""

import random
import uuid
import datetime

# ── Location pools ─────────────────────────────────────────────────────────────
HOME_CITIES = [
    ("Mumbai",    "Asia/Kolkata"),
]

NEARBY_CITIES = [
    ("Pune",      "Asia/Kolkata"),
]

FAR_CITIES = [
    ("Delhi",     "Asia/Kolkata"),
    ("Bangalore", "Asia/Kolkata"),
    ("Chennai",   "Asia/Kolkata"),
    ("Kolkata",   "Asia/Kolkata"),
    ("Hyderabad", "Asia/Kolkata"),
    ("Lagos",      "Africa/Lagos"),
    ("Singapore",  "Asia/Singapore"),
    ("Dubai",      "Asia/Dubai"),
]

# ── Device pools ───────────────────────────────────────────────────────────────
KNOWN_DEVICES   = [f"DEV-TRUSTED-{i:02d}" for i in range(1, 9)]
UNKNOWN_DEVICES = [f"NEW-DEVICE-{uuid.uuid4().hex[:6].upper()}" for _ in range(20)]

# ── Merchant pools by risk ─────────────────────────────────────────────────────
SAFE_MERCHANTS = [
    ("MERCH-GROCERY-01", "groceries"),
    ("MERCH-PHARMACY-02", "pharmacy"),
    ("MERCH-COFFEE-03", "restaurants"),
    ("MERCH-FUEL-04", "fuel"),
    ("MERCH-RETAIL-05", "retail"),
    ("MERCH-STREAMING-06", "entertainment"),
    ("MERCH-BOOK-07", "books"),
    ("MERCH-TRAVEL-08", "travel"),
]

MFA_MERCHANTS = [
    ("MERCH-ELECTRONICS-10", "electronics"),
    ("MERCH-LUXURY-11", "luxury_goods"),
    ("MERCH-FOREX-12", "forex"),
    ("MERCH-DIGITAL-13", "digital_goods"),
    ("MERCH-INSURANCE-14", "insurance"),
    ("MERCH-AIRLINE-15", "airlines"),
]

BLOCK_MERCHANTS = [
    ("MERCH-CRYPTO-20", "cryptocurrency"),
    ("MERCH-WIRE-21", "wire_transfer"),
    ("MERCH-CASINO-22", "casino"),
    ("MERCH-GAMBLING-23", "gambling"),
    ("MERCH-UNKNOWN-24", "unknown"),
    ("MERCH-HIEND-25", "high-risk"),
]

# ── TX type pools ──────────────────────────────────────────────────────────────
SAFE_TX_TYPES  = ["PAYMENT", "DEBIT"]
RISKY_TX_TYPES = ["TRANSFER", "CASH_OUT"]

# ── User pool (realistic IDs) ──────────────────────────────────────────────────
USERS = [f"user_{n:04d}" for n in range(1001, 1051)]

CHANNELS = ["mobile", "web", "api", "pos"]


def _make_safe_tx(tx_id: str) -> dict:
    """Low amount · known device · home location → expected APPROVE"""
    user_id  = random.choice(USERS)
    city, tz = random.choice(HOME_CITIES)
    merch_id, merch_cat = random.choice(SAFE_MERCHANTS)
    device   = random.choice(KNOWN_DEVICES)

    case = random.choice(["A", "B", "C", "D"])
    if case == "A":
        amount = round(random.uniform(10, 200), 2)
    elif case == "B":
        amount = round(random.uniform(500, 3000), 2)
    elif case == "C":
        amount = round(random.uniform(10000, 25000), 2)
    else:  # CASE D: repeated small
        amount = round(random.uniform(50, 500), 2)

    balance  = round(random.uniform(max(5000, amount * 2), 50000), 2)
    timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 3600))).isoformat()

    return {
        "tx_id":             tx_id,
        "user_id":           user_id,
        "amount":            amount,
        "merchant_id":       merch_id,
        "merchant_category": merch_cat,
        "device_id":         device,
        "city":              city,
        "device_timezone":   tz,
        "timestamp":         timestamp,
        "channel":           random.choice(CHANNELS),
        "oldbalanceOrg":     balance,
        "newbalanceOrig":    round(balance - amount, 2),
        "oldbalanceDest":    0.0,
        "newbalanceDest":    amount,
        "tx_type":           "PAYMENT",
        "_profile":          "SAFE",
    }


def _make_mfa_tx(tx_id: str) -> dict:
    """Medium amount · foreign location or risky merchant → expected MFA_HOLD"""
    user_id  = random.choice(USERS)
    
    if random.random() < 0.5:
        amount = round(random.uniform(1000, 5000), 2)
    else:
        amount = round(random.uniform(5000, 15000), 2)

    balance  = round(random.uniform(max(10000, amount * 1.5), 30000), 2)

    if random.random() < 0.5:
        city, tz = random.choice(NEARBY_CITIES)
    else:
        city, tz = random.choice(FAR_CITIES)

    if random.random() < 0.5:
        merch_id, merch_cat = random.choice(SAFE_MERCHANTS)
    else:
        merch_id, merch_cat = random.choice(MFA_MERCHANTS)

    is_new = random.random() < 0.3
    device = random.choice(UNKNOWN_DEVICES) if is_new else random.choice(KNOWN_DEVICES)

    if amount > 10000 and is_new:
        merch_id, merch_cat = random.choice(MFA_MERCHANTS)
        city, tz = random.choice(HOME_CITIES)

    timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 3600))).isoformat()

    return {
        "tx_id":             tx_id,
        "user_id":           user_id,
        "amount":            amount,
        "merchant_id":       merch_id,
        "merchant_category": merch_cat,
        "device_id":         device,
        "city":              city,
        "device_timezone":   tz,
        "timestamp":         timestamp,
        "channel":           random.choice(CHANNELS),
        "oldbalanceOrg":     balance,
        "newbalanceOrig":    round(balance - amount, 2),
        "oldbalanceDest":    0.0,
        "newbalanceDest":    amount,
        "tx_type":           random.choice(SAFE_TX_TYPES + RISKY_TX_TYPES),
        "_profile":          "MFA",
    }


def _make_block_tx(tx_id: str) -> dict:
    """High amount · new device · risky merchant + foreign → expected BLOCK"""
    user_id  = random.choice(USERS)
    
    # 🔥 RANDOM HIGH AMOUNT FOR BLOCK CASE
    amount = round(random.randint(200000, 500000), 2)
    
    if not amount or amount <= 0:
        amount = random.randint(200000, 500000)

    city, tz = random.choice(FAR_CITIES)
    device = random.choice(UNKNOWN_DEVICES)
    channel = random.choice(["web", "api", "mobile"])
    
    # Force high risk merchants
    crypto_gambling = [m for m in BLOCK_MERCHANTS if m[1] in ["cryptocurrency", "casino", "gambling", "wire_transfer"]]
    merch_id, merch_cat = random.choice(crypto_gambling if crypto_gambling else BLOCK_MERCHANTS)

    balance  = round(random.uniform(amount * 1.1, amount + 500000), 2)
    timestamp = (datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 3600))).isoformat()

    return {
        "tx_id":             tx_id,
        "user_id":           user_id,
        "amount":            amount,
        "merchant_id":       merch_id,
        "merchant_category": merch_cat,
        "device_id":         device,
        "city":              city,
        "device_timezone":   tz,
        "timestamp":         timestamp,
        "channel":           channel,
        "oldbalanceOrg":     balance,
        "newbalanceOrig":    round(max(0, balance - amount), 2),
        "oldbalanceDest":    0.0,
        "newbalanceDest":    amount,
        "tx_type":           random.choice(RISKY_TX_TYPES),
        "_profile":          "BLOCK",
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_batch(n: int = 10) -> list[dict]:
    """
    Generate n transactions with distribution:
      40% SAFE · 35% MFA · 25% BLOCK
    All tx_ids are unique UUIDs.
    """
    txs = []
    for _ in range(n):
        tx_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
        roll  = random.random()

        if roll < 0.40:
            txs.append(_make_safe_tx(tx_id))
        elif roll < 0.75:       # 0.40 + 0.35
            txs.append(_make_mfa_tx(tx_id))
        else:                   # top 25%
            txs.append(_make_block_tx(tx_id))

    return txs


def generate_single(profile: str = "random") -> dict:
    """
    Generate a single transaction.
    profile: 'safe' | 'mfa' | 'block' | 'random'
    """
    tx_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"

    p = profile.lower()
    if p == "safe":
        return _make_safe_tx(tx_id)
    elif p == "mfa":
        return _make_mfa_tx(tx_id)
    elif p == "block":
        return _make_block_tx(tx_id)
    else:
        return generate_batch(1)[0]
