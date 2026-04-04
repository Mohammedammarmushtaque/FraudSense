"""
backend/services/risk_engine.py
FraudSense – Structured Risk Scoring Pipeline  (REWRITTEN v4)

Risk score is now guaranteed 0-100, fully explainable, and properly
escalates for high-risk situations (INR + gambling, large amounts, etc.)

Four primary rule layers:
  1. Transaction Amount Layer    — aggressive amount tiers, high-risk category boost, profile deviation
  2. Location Anomaly Layer      — timezone mismatch + impossible travel
  3. Device Mismatch Layer       — new/untrusted device + velocity burst
  4. Behavioral Deviation Layer  — stealth probing + hour deviation

Two secondary ML / graph layers:
  5. ML Anomaly Layer            — Isolation Forest + XGBoost ensemble (booster only)
  6. Graph / Chain Layer         — network topology + state machine

FINAL SCORE FORMULA
  rule_score   = 0.6 × weighted_avg + 0.4 × max_component (no compression)
  final_score  = rule_score + ML boost (if ml_prob > 0.7)
  clamped to [0, 100]

DECISION THRESHOLDS  (0-100 scale)
  ≥ 75  → BLOCK
  ≥ 45  → MFA_HOLD
  < 45  → APPROVE

HIGH-RISK ESCALATION RULES (override / boost)
  • Currency INR + gambling/casino/forex/crypto → +40 pts on amount layer
  • Any single layer ≥ 80 → ensure final ≥ 60 (MFA at minimum)
  • Two or more layers ≥ 60 → ensure final ≥ 75 (BLOCK)
"""

from __future__ import annotations

from backend.services.device_check import (
    check_device,
    check_timezone_mismatch,
    check_impossible_travel,
    check_velocity,
)
from backend.services.behavioral import compute_behavioral_deviation, check_stealth_signatures
from backend.services.ml_service import predict_fraud, explain_prediction
from backend.services.scoring import calculate_final_risk_score


# ── Layer weights (must sum to 1.0) ───────────────────────────────────────────
LAYER_WEIGHTS = {
    "amount":     0.25,
    "location":   0.15,
    "device":     0.15,
    "behavioral": 0.20,
    "chain":      0.10,
    "graph":      0.15,
}

# Categories that trigger high-risk boost in amount scoring
HIGH_RISK_AMOUNT_CATEGORIES = {
    "gambling", "casino", "betting", "crypto", "forex",
}

# Extended high-risk categories for other layers
HIGH_RISK_CATEGORIES = {
    "cryptocurrency", "crypto", "bitcoin",
    "wire_transfer", "wire",
    "gambling", "casino", "betting",
    "forex", "fx",
    "adult", "dark_market",
}

# Currencies that amplify high-risk category penalties
HIGH_RISK_CURRENCIES = {"INR", "BTC", "ETH", "USDT"}

# Decision thresholds (0-100 scale)
BLOCK_THRESHOLD = 75
MFA_THRESHOLD   = 45


# ═════════════════════════════════════════════════════════════════════════════
class RiskEngine:
    def __init__(self, db, profile_service, graph_service):
        self.db = db
        self.profile_service = profile_service
        self.graph_service = graph_service

    # ─────────────────────────────────────────────────────────────────────────
    def calculate_risk(self, user_id: str, tx_data: dict) -> dict:
        """
        Compute a 0-100 composite fraud risk score with per-layer breakdowns.

        Returns:
            {
                "risk_score":        int   (0–100),
                "reasons":           list[str],
                "component_scores":  dict  (per-layer 0-100 scores),
                "decision":          str   (APPROVE / MFA_HOLD / BLOCK),
            }
        """
        reasons: list[str] = []
        component_scores: dict = {}

        # ── Shared context ─────────────────────────────────────────────────
        profile = self.profile_service.get_profile(user_id)
        cursor  = self.db.cursor()
        cursor.execute(
            "SELECT amount FROM transactions WHERE user_id=? ORDER BY timestamp DESC LIMIT 10",
            (user_id,),
        )
        recent_txs = [{"amount": r["amount"]} for r in cursor.fetchall()]

        currency = tx_data.get("currency", "INR").upper()
        category = tx_data.get("merchant_category", "").lower().strip()

        # ── 1. AMOUNT LAYER (0-100) ───────────────────────────────────────
        amount_score, amount_reasons = self._score_amount(tx_data, profile, currency, category)
        component_scores["amount"] = round(amount_score, 1)
        reasons.extend(amount_reasons)

        # ── 2. LOCATION LAYER (0-100) ─────────────────────────────────────
        city = tx_data.get("city", "unknown_city")
        tz   = tx_data.get("device_timezone", "unknown")

        loc_score = 0.0
        tz_chk = check_timezone_mismatch(user_id, tz, city, self.db)
        if tz_chk["risk_add"] > 0:
            loc_score += 30
            reasons.append("VPN or Timezone Mismatch detected")

        imp_chk = check_impossible_travel(user_id, city, tx_data.get("timestamp"), self.db)
        if imp_chk["risk_add"] > 0:
            loc_score += 40
            reasons.append("Impossible Travel detected")

        component_scores["location"] = round(min(100.0, loc_score), 1)

        # ── 3. DEVICE LAYER (0-100) ───────────────────────────────────────
        device_id = tx_data.get("device_id", "unknown_device")
        dev_chk   = check_device(user_id, device_id, self.db)
        dev_score = 0.0

        if dev_chk["risk_add"] > 0:
            dev_score += 30
            reasons.append("New or untrusted device detected")

        tx_data["velocity_count_1h"] = len(recent_txs)
        vel_chk = check_velocity(user_id, tx_data.get("timestamp"), self.db)
        if vel_chk["risk_add"] > 0:
            dev_score += 25
            reasons.append("High transaction velocity — velocity breach")

        component_scores["device"] = round(min(100.0, dev_score), 1)

        chain_event = dev_chk.get("chain_event")
        if chain_event:
            self.graph_service.process_chain_event(user_id, chain_event)

        # 🔥 COMBINATION BOOST: Location + Device anomaly
        if component_scores.get("location", 0) > 50 and component_scores.get("device", 0) > 40:
            component_scores["location"] = min(100, component_scores["location"] + 20)
            component_scores["device"] = min(100, component_scores["device"] + 20)
            reasons.append("Location + device anomaly escalation")

        # ── 4. BEHAVIORAL LAYER (0-100) ───────────────────────────────────
        beh_res = compute_behavioral_deviation(tx_data, profile)
        beh_raw = beh_res["deviation_score"]          # 0-70 from behavioral module
        beh_score = min(100.0, (beh_raw / 0.70))      # normalize to 0-100
        reasons.extend(beh_res["signals"])

        stealth = check_stealth_signatures(tx_data, recent_txs)
        if stealth["risk_add"] > 0:
            beh_score = min(100.0, beh_score + stealth["risk_add"])
            reasons.append(stealth["label"])

        component_scores["behavioral"] = round(beh_score, 1)

        # ── 5. ML LAYER (0-100, stored separately for boost) ─────────────
        prediction_result = predict_fraud(tx_data)
        ml_prob  = prediction_result["fraud_probability"]      # 0-1
        ml_score = prediction_result["ml_risk_score"]          # 0-100
        ml_explanations = explain_prediction(prediction_result["features"])

        component_scores["ml"]                = round(ml_score, 1)
        component_scores["_fraud_probability"] = ml_prob
        component_scores["_ml_explanations"]   = ml_explanations

        if ml_prob > 0.6:
            reasons.append(f"ML model flagged as fraud — confidence {ml_prob:.0%}")
        for expl in ml_explanations:
            reasons.append(f"ML signal: {expl}")

        # ── 6. GRAPH / CHAIN LAYER (0-100) ────────────────────────────────
        merchant_id = tx_data.get("merchant_id")
        graph_res   = self.graph_service.detect_suspicious_clusters(user_id, merchant_id)
        chain_boost = self.graph_service.get_chain_risk_boost(user_id)

        if graph_res["graph_risk"] > 0:
            reasons.extend(graph_res["reasons"])
        if chain_boost > 0:
            reasons.append("Suspicious chain sequence pattern detected")

        component_scores["graph"] = round(min(100.0, graph_res["graph_risk"]), 1)
        component_scores["chain"] = round(min(100.0, chain_boost), 1)

        # ── RULE SCORE (weighted combination of layers 1-4 + 6) ──────────
        rule_input = {
            "amount":     component_scores["amount"],
            "location":   component_scores["location"],
            "device":     component_scores["device"],
            "behavioral": component_scores["behavioral"],
            "graph":      component_scores["graph"],
            "chain":      component_scores["chain"],
        }

        # 🔥 HYBRID RULE SCORING (NO MORE COMPRESSION)
        base_score = calculate_final_risk_score(rule_input, LAYER_WEIGHTS)

        max_component = max(rule_input.values())

        # Combine average + strongest signal
        rule_score_100 = (0.6 * base_score) + (0.4 * max_component)

        # 🔥 MULTI-SIGNAL BOOST
        high_components = sum(1 for v in rule_input.values() if v > 60)

        if high_components >= 4:
            rule_score_100 += 35
        elif high_components >= 3:
            rule_score_100 += 25
        elif high_components >= 2:
            rule_score_100 += 15

        rule_score_100 = min(100, rule_score_100)

        # ── BLEND: DOMINANT RULE-BASED + ML BOOSTER ───────────────────────
        # 🔥 STEP 1: BASE + ML BOOST
        final_score = rule_score_100

        if ml_prob > 0.85:
            final_score += 15
        elif ml_prob > 0.7:
            final_score += 10

        # 🔥 STEP 2: EXPAND SCORE RANGE
        if final_score > 50:
            final_score += 10

        if final_score > 65:
            final_score += 10

        if final_score > 80:
            final_score += 5

        # 🔥 STEP 3: FINAL CLAMP
        final_score = min(100, final_score)

        # ── HIGH-RISK ESCALATION OVERRIDES ────────────────────────────────
        final_score = self._apply_escalation_rules(
            final_score, component_scores, currency, category, reasons
        )

        # 🔥 HARD FRAUD FLOOR
        if (
            component_scores.get("amount", 0) > 70 and
            component_scores.get("device", 0) > 40
        ):
            final_score = max(final_score, 80)
            reasons.append("High-confidence fraud pattern detected")

        if (
            component_scores.get("amount", 0) > 80 and
            component_scores.get("behavioral", 0) > 70
        ):
            final_score = max(final_score, 90)
            reasons.append("High-confidence fraud pattern detected")

        # 🔥 CONTEXT-AWARE AMOUNT LOGIC
        amount_score = component_scores.get("amount", 0)
        device_score = component_scores.get("device", 0)
        location_score = component_scores.get("location", 0)
        merchant = tx_data.get("merchant_category", "").lower()

        # Define safe high-value categories
        SAFE_HIGH_VALUE_CATEGORIES = [
            "healthcare",
            "insurance",
            "education",
            "travel",
            "hospital"
        ]
        is_safe_category = any(cat in merchant for cat in SAFE_HIGH_VALUE_CATEGORIES)

        # 🔹 CASE 1 — SAFE CONTEXT (High amount but normal behavior)
        if amount_score >= 80 and is_safe_category and device_score < 30 and location_score < 30:
            final_score = max(final_score, 65)
            reasons.append("High-value but trusted transaction (safe context)")

        # 🔹 CASE 2 — MODERATE RISK (MFA)
        elif amount_score >= 80 and (device_score < 50 and location_score < 50):
            final_score = max(final_score, 75)
            reasons.append("High-value transaction requires verification")

        # 🔹 CASE 3 — HIGH RISK (BLOCK)
        elif amount_score >= 80:
            final_score = max(final_score, 90)
            reasons.append("High-value suspicious transaction")

        final_score = int(round(min(100.0, max(0.0, final_score))))

        # ── DECISION ──────────────────────────────────────────────────────
        if final_score >= BLOCK_THRESHOLD:
            decision = "BLOCK"
        elif final_score >= MFA_THRESHOLD:
            decision = "MFA_HOLD"
        else:
            decision = "APPROVE"

        # Deduplicate reasons, preserve order
        unique_reasons = list(dict.fromkeys(r for r in reasons if r))

        # Add decision-context reason at top
        if decision == "BLOCK":
            unique_reasons.insert(0, f"Transaction BLOCKED — risk score {final_score}/100 exceeds safety threshold")
        elif decision == "MFA_HOLD":
            unique_reasons.insert(0, f"MFA required — risk score {final_score}/100 above alert threshold")

        return {
            "risk_score":       final_score,
            "decision":         decision,
            "reasons":          unique_reasons,
            "component_scores": component_scores,
            "rule_score":       round(rule_score_100, 1),
            "ml_score":         round(ml_score, 1),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _score_amount(
        self,
        tx_data: dict,
        profile: dict,
        currency: str,
        category: str,
    ) -> tuple[float, list[str]]:
        """
        Returns (score: float 0-100, reasons: list[str]).

        Aggressive amount scoring designed to produce scores up to 95-100
        for high-value, high-risk transactions.

        Scoring layers:
          1. Base amount scaling (tiered)
          2. High-risk category boost (gambling, casino, betting, crypto, forex)
          3. Profile deviation boost (vs user's avg_amount)
          4. Final cap at 100
        """
        score   = 0.0
        reasons = []

        amount = float(tx_data.get("amount", 0))

        # ══════════════════════════════════════════════════════════════════
        # 1. BASE AMOUNT SCALING
        # ══════════════════════════════════════════════════════════════════
        if amount < 1000:
            score = 5
        elif amount < 10000:
            score = 15
            reasons.append("Moderate transaction amount detected")
        elif amount < 50000:
            score = 35
            reasons.append("High transaction amount detected")
        elif amount < 80000:
            score = 60
            reasons.append("High transaction amount detected")
        else:
            score = 80
            reasons.append("High transaction amount detected")

        # ══════════════════════════════════════════════════════════════════
        # 2. HIGH-RISK CATEGORY BOOST
        # ══════════════════════════════════════════════════════════════════
        is_high_risk_cat = any(
            high_risk_kw in category
            for high_risk_kw in HIGH_RISK_AMOUNT_CATEGORIES
        )

        if is_high_risk_cat:
            reasons.append("High-risk merchant category")

            if amount > 50000:
                score = max(score, 90)
            elif amount > 20000:
                score += 20

        # ══════════════════════════════════════════════════════════════════
        # 3. PROFILE DEVIATION BOOST
        # ══════════════════════════════════════════════════════════════════
        avg_amount = profile.get("avg_amount")

        if avg_amount is not None and avg_amount > 0:
            avg_amount = float(avg_amount)

            if amount > avg_amount * 5:
                score += 25
                reasons.append("Transaction significantly exceeds user average")
            elif amount > avg_amount * 3:
                score += 15
                reasons.append("Transaction significantly exceeds user average")

        # ══════════════════════════════════════════════════════════════════
        # 4. FINAL CAP
        # ══════════════════════════════════════════════════════════════════
        score = min(100, score)

        return score, reasons

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_escalation_rules(
        score: float,
        cs: dict,
        currency: str,
        category: str,
        reasons: list[str],
    ) -> float:
        """
        Post-blend escalation:
          • Any single layer ≥ 80 → floor at 60 (MFA guaranteed)
          • Two or more layers ≥ 60 → floor at 75 (BLOCK guaranteed)
          • Currency+Category critical combo → +15 pts
          • Extreme fraud pattern (amount + behavioral) → floor at 85
        """
        layer_scores = [
            cs.get("amount", 0),
            cs.get("location", 0),
            cs.get("device", 0),
            cs.get("behavioral", 0),
            cs.get("graph", 0),
            cs.get("chain", 0),
        ]

        # Escalation 1 — any extremely hot layer
        hot_layers = [s for s in layer_scores if s >= 80]
        if hot_layers:
            score = max(score, 60.0)

        # Escalation 2 — multiple elevated layers = compound fraud signal
        elevated = [s for s in layer_scores if s >= 60]
        if len(elevated) >= 2:
            score = max(score, 75.0)
            reasons.append(
                f"Multiple risk layers elevated ({len(elevated)} layers ≥60) — compound fraud signal"
            )

        # Escalation 3 — critical currency + category combo
        is_high_risk_cat = any(h in category for h in HIGH_RISK_CATEGORIES)
        if currency in HIGH_RISK_CURRENCIES and is_high_risk_cat:
            score = min(100.0, score + 15)

        # 🔥 EXTREME FRAUD ESCALATION
        if cs.get("amount", 0) > 80 and cs.get("behavioral", 0) > 70:
            score = max(score, 85)
            reasons.append("Extreme fraud pattern detected (amount + behavioral anomaly)")

        return score


# ─────────────────────────────────────────────────────────────────────────────
def _fmt(amount: float) -> str:
    """Format amount for reason strings."""
    if amount >= 100_000:
        return f"₹{amount/100_000:.1f}L"
    if amount >= 1_000:
        return f"₹{amount:,.0f}"
    return f"₹{amount:.0f}"