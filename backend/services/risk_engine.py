"""
backend/services/risk_engine.py
FraudSense – Structured Risk Scoring Pipeline (FIXED v5)

Risk score now properly spans 0-100 with clear differentiation:
- Low-risk transactions: < 30
- High-risk transactions: > 70

Key fix: Removed averaging compression. Now uses dominant-signal scoring
where the highest risk component drives the score, with other components
adding weighted contributions. No step-based expansion that only applies
to already-high scores.

DECISION THRESHOLDS (0-100 scale)
  ≥ 75  → BLOCK
  ≥ 45  → MFA_HOLD
  < 45  → APPROVE
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


# ── Layer weights for contribution scoring ───────────────────────────────────────
LAYER_WEIGHTS = {
    "amount":     0.30,
    "location":   0.15,
    "device":     0.15,
    "behavioral": 0.20,
    "chain":      0.10,
    "graph":      0.10,
}

# Categories that trigger high-risk boost
HIGH_RISK_AMOUNT_CATEGORIES = {
    "gambling", "casino", "betting", "crypto", "forex",
}

HIGH_RISK_CATEGORIES = {
    "cryptocurrency", "crypto", "bitcoin",
    "wire_transfer", "wire",
    "gambling", "casino", "betting",
    "forex", "fx",
    "adult", "dark_market",
}

HIGH_RISK_CURRENCIES = {"INR", "BTC", "ETH", "USDT"}

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

        # ══════════════════════════════════════════════════════════════════
        # COMPUTE EACH LAYER (0-100 scale per layer)
        # ══════════════════════════════════════════════════════════════════

        # ── 1. AMOUNT LAYER ───────────────────────────────────────────────
        amount_score, amount_reasons = self._score_amount(tx_data, profile, currency, category)
        component_scores["amount"] = round(amount_score, 1)
        reasons.extend(amount_reasons)

        # ── 2. LOCATION LAYER ─────────────────────────────────────────────
        city = tx_data.get("city", "unknown_city")
        tz   = tx_data.get("device_timezone", "unknown")

        loc_score = 0.0
        tz_chk = check_timezone_mismatch(user_id, tz, city, self.db)
        if tz_chk["risk_add"] > 0:
            loc_score += 35
            reasons.append("VPN or Timezone Mismatch detected")

        imp_chk = check_impossible_travel(user_id, city, tx_data.get("timestamp"), self.db)
        if imp_chk["risk_add"] > 0:
            loc_score += 50
            reasons.append("Impossible Travel detected")

        component_scores["location"] = round(min(100.0, loc_score), 1)

        # ── 3. DEVICE LAYER ───────────────────────────────────────────────
        device_id = tx_data.get("device_id", "unknown_device")
        dev_chk   = check_device(user_id, device_id, self.db)
        dev_score = 0.0

        if dev_chk["risk_add"] > 0:
            dev_score += 40
            reasons.append("New or untrusted device detected")

        tx_data["velocity_count_1h"] = len(recent_txs)
        vel_chk = check_velocity(user_id, tx_data.get("timestamp"), self.db)
        if vel_chk["risk_add"] > 0:
            dev_score += 35
            reasons.append("High transaction velocity — velocity breach")

        component_scores["device"] = round(min(100.0, dev_score), 1)

        chain_event = dev_chk.get("chain_event")
        if chain_event:
            self.graph_service.process_chain_event(user_id, chain_event)

        # Combination boost
        if component_scores.get("location", 0) > 50 and component_scores.get("device", 0) > 40:
            component_scores["location"] = min(100, component_scores["location"] + 15)
            component_scores["device"] = min(100, component_scores["device"] + 15)
            reasons.append("Location + device anomaly escalation")

        # ── 4. BEHAVIORAL LAYER ───────────────────────────────────────────
        beh_res = compute_behavioral_deviation(tx_data, profile)
        beh_raw = beh_res["deviation_score"]
        beh_score = min(100.0, beh_raw / 0.70) if beh_raw > 0 else 0.0
        reasons.extend(beh_res["signals"])

        stealth = check_stealth_signatures(tx_data, recent_txs)
        if stealth["risk_add"] > 0:
            beh_score = min(100.0, beh_score + stealth["risk_add"])
            reasons.append(stealth["label"])

        component_scores["behavioral"] = round(beh_score, 1)

        # ── 5. ML LAYER ───────────────────────────────────────────────────
        prediction_result = predict_fraud(tx_data)
        ml_prob  = prediction_result["fraud_probability"]
        ml_score = prediction_result["ml_risk_score"]
        ml_explanations = explain_prediction(prediction_result["features"])

        component_scores["ml"] = round(ml_score, 1)
        component_scores["_fraud_probability"] = ml_prob
        component_scores["_ml_explanations"] = ml_explanations

        if ml_prob > 0.6:
            reasons.append(f"ML model flagged as fraud — confidence {ml_prob:.0%}")
        for expl in ml_explanations:
            reasons.append(f"ML signal: {expl}")

        # ── 6. GRAPH / CHAIN LAYER ────────────────────────────────────────
        merchant_id = tx_data.get("merchant_id")
        graph_res   = self.graph_service.detect_suspicious_clusters(user_id, merchant_id)
        chain_boost = self.graph_service.get_chain_risk_boost(user_id)

        if graph_res["graph_risk"] > 0:
            reasons.extend(graph_res["reasons"])
        if chain_boost > 0:
            reasons.append("Suspicious chain sequence pattern detected")

        component_scores["graph"] = round(min(100.0, graph_res["graph_risk"]), 1)
        component_scores["chain"] = round(min(100.0, chain_boost), 1)

        # ══════════════════════════════════════════════════════════════════
        # FINAL SCORE CALCULATION (NO COMPRESSION)
        # ══════════════════════════════════════════════════════════════════

        rule_layers = {
            "amount":     component_scores["amount"],
            "location":   component_scores["location"],
            "device":     component_scores["device"],
            "behavioral": component_scores["behavioral"],
            "graph":      component_scores["graph"],
            "chain":      component_scores["chain"],
        }

        # 🔥 DOMINANT SIGNAL SCORING (fixed weights, no averaging compression)
        # The highest component becomes the base, others add weighted contributions
        max_score = max(rule_layers.values())
        max_layer = max(rule_layers, key=rule_layers.get)

        # Calculate weighted contribution from all layers
        weighted_sum = 0.0
        for layer, score in rule_layers.items():
            weighted_sum += score * LAYER_WEIGHTS[layer]

        # 🔥 KEY FIX: Use MAX as dominant, weighted sum as secondary contribution
        # This ensures high values stay high (not averaged down)
        # and low values stay low (not averaged up)
        # - Dominant signal (max): contributes 70% of final
        # - Weighted average: contributes 30% of final
        # This preserves the full 0-100 range
        final_score = (0.70 * max_score) + (0.30 * weighted_sum)

        # ── MULTI-SIGNAL ESCALATION ───────────────────────────────────────
        high_components = sum(1 for v in rule_layers.values() if v >= 60)
        medium_components = sum(1 for v in rule_layers.values() if v >= 40)

        # Multiple elevated signals = stronger fraud indication
        if high_components >= 3:
            final_score = max(final_score, 85.0)
            reasons.append("Multiple high-risk indicators detected (≥3 layers ≥60)")
        elif high_components >= 2:
            final_score = max(final_score, 75.0)
            reasons.append("Multiple risk layers elevated (≥2 layers ≥60)")
        elif medium_components >= 3:
            final_score = max(final_score, 55.0)

        # ── ML BOOST ───────────────────────────────────────────────────────
        if ml_prob > 0.85:
            final_score += 20
        elif ml_prob > 0.70:
            final_score += 15
        elif ml_prob > 0.55:
            final_score += 8

        # ── CRITICAL PATTERN ESCALATIONS ───────────────────────────────────
        # Amount + Device anomaly
        if rule_layers["amount"] > 70 and rule_layers["device"] > 40:
            final_score = max(final_score, 80.0)
            reasons.append("High-confidence fraud pattern detected (amount + device)")

        # Amount + Behavioral
        if rule_layers["amount"] > 75 and rule_layers["behavioral"] > 65:
            final_score = max(final_score, 85.0)
            reasons.append("High-confidence fraud pattern detected (amount + behavioral)")

        # Location + Device
        if rule_layers["location"] > 60 and rule_layers["device"] > 50:
            final_score = max(final_score, 75.0)
            reasons.append("Location and device anomaly combination")

        # ── HIGH-RISK CATEGORY + CURRENCY BOOST ─────────────────────────────
        is_high_risk_cat = any(h in category for h in HIGH_RISK_CATEGORIES)
        if currency in HIGH_RISK_CURRENCIES and is_high_risk_cat:
            final_score = min(100.0, final_score + 20)
            reasons.append("High-risk category with flagged currency")

        # ── CONTEXT-AWARE AMOUNT ADJUSTMENTS ───────────────────────────────
        merchant = tx_data.get("merchant_category", "").lower()

        SAFE_HIGH_VALUE_CATEGORIES = ["healthcare", "insurance", "education", "travel", "hospital"]
        is_safe_category = any(cat in merchant for cat in SAFE_HIGH_VALUE_CATEGORIES)

        # Safe context: high amount but all other signals clean → reduce score
        if rule_layers["amount"] >= 70 and is_safe_category:
            if rule_layers["device"] < 25 and rule_layers["location"] < 25 and rule_layers["behavioral"] < 25:
                final_score = min(final_score, 50.0)
                reasons.append("High-value but trusted transaction (safe context)")

        # ── FINAL CLAMP ────────────────────────────────────────────────────
        final_score = int(round(min(100.0, max(0.0, final_score))))

        # ── DECISION ────────────────────────────────────────────────────────
        if final_score >= BLOCK_THRESHOLD:
            decision = "BLOCK"
        elif final_score >= MFA_THRESHOLD:
            decision = "MFA_HOLD"
        else:
            decision = "APPROVE"

        # Deduplicate reasons
        unique_reasons = list(dict.fromkeys(r for r in reasons if r))

        # Add decision context
        if decision == "BLOCK":
            unique_reasons.insert(0, f"Transaction BLOCKED — risk score {final_score}/100 exceeds safety threshold")
        elif decision == "MFA_HOLD":
            unique_reasons.insert(0, f"MFA required — risk score {final_score}/100 above alert threshold")

        return {
            "risk_score":       final_score,
            "decision":         decision,
            "reasons":          unique_reasons,
            "component_scores": component_scores,
            "rule_score":       round(final_score, 1),
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

        Direct tiered scoring without compression:
        - Low amounts (< 1000): score 5
        - Moderate (1000-10000): score 15
        - High (10000-50000): score 40
        - Very high (50000-100000): score 65
        - Extreme (> 100000): score 85

        Plus category and deviation boosts.
        """
        score   = 0.0
        reasons = []

        amount = float(tx_data.get("amount", 0))

        # ══════════════════════════════════════════════════════════════════
        # 1. BASE AMOUNT SCORING (direct tiers, no normalization)
        # ══════════════════════════════════════════════════════════════════
        if amount < 1000:
            score = 5
        elif amount < 10000:
            score = 15
            reasons.append("Moderate transaction amount")
        elif amount < 50000:
            score = 40
            reasons.append("High transaction amount")
        elif amount < 100000:
            score = 65
            reasons.append("Very high transaction amount")
        else:
            score = 85
            reasons.append("Extreme transaction amount")

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
                score = max(score, 95)
            elif amount > 20000:
                score += 25
            else:
                score += 15

        # ══════════════════════════════════════════════════════════════════
        # 3. PROFILE DEVIATION BOOST
        # ══════════════════════════════════════════════════════════════════
        avg_amount = profile.get("avg_amount")

        if avg_amount is not None and avg_amount > 0:
            avg_amount = float(avg_amount)

            if amount > avg_amount * 5:
                score += 30
                reasons.append("Transaction far exceeds user average")
            elif amount > avg_amount * 3:
                score += 20
                reasons.append("Transaction exceeds user average")
            elif amount > avg_amount * 2:
                score += 10
                reasons.append("Transaction above user average")

        # ══════════════════════════════════════════════════════════════════
        # 4. FINAL CAP
        # ══════════════════════════════════════════════════════════════════
        score = min(100, score)

        return score, reasons


# ─────────────────────────────────────────────────────────────────────────────
def _fmt(amount: float) -> str:
    """Format amount for reason strings."""
    if amount >= 100_000:
        return f"₹{amount/100_000:.1f}L"
    if amount >= 1_000:
        return f"₹{amount:,.0f}"
    return f"₹{amount:.0f}"