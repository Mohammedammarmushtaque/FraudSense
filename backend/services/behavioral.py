"""
backend/services/behavioral.py
FraudSense — Behavioral Profiling & Anomaly Detection

Maintains per-user transaction history for real-time anomaly detection.

Features:
  - Last 5 transactions tracked per user (SQLite + in-memory cache)
  - Running average transaction amount
  - Frequent locations list
  - Anomaly detection:
      * Amount deviation > 2× user average → high risk
      * New location not seen before → flagged
      * Unusual transaction hour → flagged
  - Returns behavioral_risk_score (0.0–1.0)

Storage:
  - Primary: SQLite user_profiles table (persistent)
  - Cache: In-memory _profile_cache dict (fast O(1) lookup per request)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from collections import deque

# ── In-memory profile cache for sub-millisecond repeated lookups ──────────────
# Structure: { user_id: { "history": deque(maxlen=5), "avg_amount": float,
#                         "locations": list, "merchants": list, "typical_hour": int } }
_profile_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

HISTORY_SIZE = 5           # last N transactions tracked
DEVIATION_THRESHOLD = 2.0  # amount > threshold × avg → anomaly


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_cached(user_id: str) -> dict | None:
    with _cache_lock:
        return _profile_cache.get(user_id)


def _update_cache(user_id: str, patch: dict) -> None:
    with _cache_lock:
        if user_id not in _profile_cache:
            _profile_cache[user_id] = {
                "history":      deque(maxlen=HISTORY_SIZE),
                "avg_amount":   0.0,
                "locations":    [],
                "merchants":    [],
                "typical_hour": 12,
            }
        _profile_cache[user_id].update(patch)


# ─────────────────────────────────────────────────────────────────────────────
# Public: get_user_profile  (DB-backed with cache warm-up)
# ─────────────────────────────────────────────────────────────────────────────

def get_user_profile(user_id: str, db) -> dict:
    """
    Return a user's behavioral profile with last-5 transaction history.

    Fast path: in-memory cache (microseconds).
    Slow path: SQLite query then cache population (first request per user).
    """
    # Fast path
    cached = _get_cached(user_id)
    if cached is not None:
        return _to_profile_dict(cached)

    # Slow path — hit DB then warm cache
    profile = _load_from_db(user_id, db)
    _update_cache(user_id, profile)
    return _to_profile_dict(_profile_cache[user_id])


def _to_profile_dict(cached: dict) -> dict:
    """Convert internal cache entry to the profile dict shape consumers expect."""
    history = list(cached.get("history", []))
    return {
        "avg_amount":       cached.get("avg_amount", 0.0),
        "std_amount":       cached.get("std_amount", 0.0),
        "common_merchants": cached.get("merchants", []),
        "frequent_locations": cached.get("locations", []),
        "typical_hour":     cached.get("typical_hour", 12),
        "tx_count":         cached.get("tx_count", 0),
        "last_5_amounts":   [t.get("amount", 0) for t in history],
        "last_5_locations": [t.get("city", "") for t in history],
    }


def _load_from_db(user_id: str, db) -> dict:
    """Load profile from user_profiles + last 5 transactions from DB."""
    cursor = db.cursor()

    # Profile row
    cursor.execute(
        "SELECT avg_amount, std_amount, common_merchants, typical_hour, tx_count "
        "FROM user_profiles WHERE user_id = ?",
        (user_id,)
    )
    row = cursor.fetchone()

    avg_amount  = float(row["avg_amount"])  if row and row["avg_amount"]  else 0.0
    std_amount  = float(row["std_amount"])  if row and row["std_amount"]  else 0.0
    typical_hour = int(row["typical_hour"]) if row and row["typical_hour"] else 12
    tx_count     = int(row["tx_count"])      if row and row["tx_count"]     else 0

    merchants_json = row["common_merchants"] if row else None
    merchants = json.loads(merchants_json) if merchants_json else []

    # Last 5 transactions for history
    cursor.execute(
        "SELECT amount, city, merchant_id, timestamp "
        "FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, HISTORY_SIZE),
    )
    rows = cursor.fetchall()
    history = deque(
        [{"amount": r["amount"], "city": r["city"], "merchant_id": r["merchant_id"]}
         for r in rows],
        maxlen=HISTORY_SIZE,
    )

    locations = list({r["city"] for r in rows if r["city"]})

    return {
        "avg_amount":   avg_amount,
        "std_amount":   std_amount,
        "typical_hour": typical_hour,
        "tx_count":     tx_count,
        "merchants":    merchants,
        "locations":    locations,
        "history":      history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: record_transaction  (call AFTER a transaction is approved/logged)
# ─────────────────────────────────────────────────────────────────────────────

def record_transaction(user_id: str, tx_data: dict) -> None:
    """
    Update the in-memory behavioral profile with a new transaction.
    Maintains rolling average, location list, and history window.
    """
    amount   = float(tx_data.get("amount", 0) or 0)
    city     = str(tx_data.get("city", "") or "")
    merchant = str(tx_data.get("merchant_id", "") or "")

    with _cache_lock:
        if user_id not in _profile_cache:
            _profile_cache[user_id] = {
                "history":      deque(maxlen=HISTORY_SIZE),
                "avg_amount":   amount,
                "std_amount":   0.0,
                "locations":    [],
                "merchants":    [],
                "typical_hour": 12,
                "tx_count":     0,
            }

        p = _profile_cache[user_id]

        # Push into history window
        p["history"].append({"amount": amount, "city": city, "merchant_id": merchant})

        # Running average (Welford's online update)
        old_count = p.get("tx_count", 0)
        new_count = old_count + 1
        old_avg   = p.get("avg_amount", 0.0)
        new_avg   = old_avg + (amount - old_avg) / new_count
        p["avg_amount"] = round(new_avg, 2)
        p["tx_count"]   = new_count

        # Update location list (keep last 10 unique)
        if city and city not in p["locations"]:
            p["locations"] = (p["locations"] + [city])[-10:]

        # Update merchant list (keep last 20 unique)
        if merchant and merchant not in p["merchants"]:
            p["merchants"] = (p["merchants"] + [merchant])[-20:]


# ─────────────────────────────────────────────────────────────────────────────
# Public: compute_behavioral_deviation  (used by RiskEngine)
# ─────────────────────────────────────────────────────────────────────────────

def compute_behavioral_deviation(tx_data: dict, user_profile: dict) -> dict:
    """
    Compute a behavioral deviation score (0–70) based on the current transaction
    vs. the user's historical profile.

    Anomaly signals:
      a) Amount deviation  > 2× avg  → HIGH signal
      b) New location not in profile → flagged
      c) Unknown merchant            → flagged
      d) Unusual hour of day         → flagged
      e) Frequency spike             → flagged

    Returns:
        {
            "deviation_score": float (0–70),
            "signals":         list[str],
            "behavioral_risk": float (0.0–1.0),
        }
    """
    signals: list[str] = []
    score = 0.0

    amount      = float(tx_data.get("amount", 0) or 0)
    merchant_id = str(tx_data.get("merchant_id", "") or "")
    city        = str(tx_data.get("city", "") or "")

    avg_amount        = float(user_profile.get("avg_amount", 0) or 0)
    std_amount        = float(user_profile.get("std_amount", 1) or 1)
    common_merchants  = user_profile.get("common_merchants", [])
    frequent_locations = user_profile.get("frequent_locations", [])
    typical_hour      = int(user_profile.get("typical_hour", 12) or 12)

    # ── a) Amount deviation ────────────────────────────────────────────────
    if avg_amount > 0:
        ratio = amount / avg_amount
        z_score = abs(amount - avg_amount) / (std_amount + 1)

        if ratio > DEVIATION_THRESHOLD * 2:        # > 4× avg
            score += 30
            signals.append("amount_deviation")
        elif ratio > DEVIATION_THRESHOLD:           # > 2× avg → core requirement
            score += 20
            signals.append("amount_deviation")
        elif z_score > 1.5:
            score += 10
            signals.append("amount_deviation")

    # ── b) New location ────────────────────────────────────────────────────
    if city and frequent_locations and city not in frequent_locations:
        score += 20
        signals.append("new_location")
    elif city and not frequent_locations:
        # No history — moderate uncertainty
        score += 5

    # ── c) Unknown merchant ────────────────────────────────────────────────
    if merchant_id and common_merchants and merchant_id not in common_merchants:
        score += 10
        signals.append("unknown_merchant")

    # ── d) Unusual hour ────────────────────────────────────────────────────
    ts = tx_data.get("timestamp")
    try:
        if isinstance(ts, str):
            hour = datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
        elif hasattr(ts, "hour"):
            hour = ts.hour
        else:
            hour = datetime.now().hour

        time_deviation = abs(hour - typical_hour)
        if time_deviation > 12:
            time_deviation = 24 - time_deviation

        time_score = min(10, time_deviation * 0.8)
        score += time_score
        if time_deviation > 6:
            signals.append("unusual_hour")

    except Exception:
        pass

    # ── e) Frequency spike ─────────────────────────────────────────────────
    recent_count = float(tx_data.get("recent_tx_count", 0) or 0)
    tx_count     = int(user_profile.get("tx_count", 0) or 0)
    expected_rate = tx_count / 30.0 if tx_count > 0 else 0.0
    if recent_count > max(3, expected_rate * 1.5):
        score += 20
        signals.append("frequency_spike")

    deviation_score  = round(min(70.0, score), 2)
    behavioral_risk  = round(deviation_score / 70.0, 4)  # normalize to 0–1

    return {
        "deviation_score":  deviation_score,
        "signals":          signals,
        "behavioral_risk":  behavioral_risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public: get_behavioral_risk_score  (standalone convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def get_behavioral_risk_score(user_id: str, tx_data: dict, db=None) -> dict:
    """
    One-call interface: loads profile (cache or DB), computes deviation.

    Returns:
        {
            "behavioral_risk": float  (0.0–1.0),
            "deviation_score": float  (0–70),
            "signals":         list[str],
            "avg_amount":      float,
            "frequent_locations": list,
        }
    """
    if db is not None:
        profile = get_user_profile(user_id, db)
    else:
        cached = _get_cached(user_id)
        profile = _to_profile_dict(cached) if cached else {
            "avg_amount": 0.0, "std_amount": 0.0, "common_merchants": [],
            "frequent_locations": [], "typical_hour": 12, "tx_count": 0,
        }

    result = compute_behavioral_deviation(tx_data, profile)
    result["avg_amount"]         = profile.get("avg_amount", 0.0)
    result["frequent_locations"] = profile.get("frequent_locations", [])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility: check_stealth_signatures (unchanged — used by RiskEngine)
# ─────────────────────────────────────────────────────────────────────────────

def check_stealth_signatures(tx_data: dict, recent_transactions: list) -> dict:
    """
    Detect stealth fraud patterns from transaction sequence.

    Patterns:
      1. JUST_BELOW_THRESHOLD — amount 8,500–9,999 (structuring)
      2. MICRO_TEST_TX        — tiny amount followed by large historical txn
      3. GRADUATED_ESCALATION — systematic 15–40% step-up pattern
    """
    amount = float(tx_data.get("amount", 0) or 0)

    # Pattern 1
    if 8500 <= amount <= 9999:
        return {
            "flag":     "JUST_BELOW_THRESHOLD",
            "risk_add": 20,
            "label":    "Amount clustering just below ₹10,000 reporting threshold",
        }

    # Pattern 2
    if amount <= 5:
        for recent in recent_transactions:
            if float(recent.get("amount", 0) or 0) > 5000:
                return {
                    "flag":     "MICRO_TEST_TX",
                    "risk_add": 35,
                    "label":    "Micro test transaction preceding high-value activity",
                }

    # Pattern 3
    if len(recent_transactions) >= 3:
        amounts = [float(t.get("amount", 0) or 0) for t in recent_transactions[-4:]] + [amount]
        is_graduated = all(
            1.15 <= amounts[i] / amounts[i - 1] <= 1.40
            for i in range(1, len(amounts))
            if amounts[i - 1] > 0
        )
        if is_graduated and len(amounts) >= 4:
            return {
                "flag":     "GRADUATED_ESCALATION",
                "risk_add": 25,
                "label":    "Graduated amount escalation — systematic threshold probing",
            }

    return {"flag": None, "risk_add": 0}