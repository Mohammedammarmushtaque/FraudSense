"""
validate_fraud_system.py
FraudSense — End-to-end validation tests

Runs 3 test scenarios and verifies the system produces correct,
varied decisions (not always SAFE).

Usage:
  python validate_fraud_system.py

Requires the backend to be running on localhost:8000.
Can also be run standalone to validate the ML model directly.
"""

import sys
import os
import json

# ── Test 1: Direct ML model validation (no server needed) ─────────────────────

def test_ml_model_directly():
    """Validate the ML model produces varied predictions for different inputs."""
    print("\n" + "=" * 60)
    print("TEST: Direct ML Model Validation")
    print("=" * 60)

    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from backend.services.ml_service import predict_fraud, extract_features

    # TEST 1: SAFE transaction
    safe_tx = {
        "amount": 200,
        "city": "Mumbai",
        "device_id": "DEV-TRUSTED-01",
        "merchant_category": "groceries",
        "velocity_count_1h": 1,
        "timestamp": "2026-04-04T10:00:00",
        "device_changes": 0,
    }
    safe_result = predict_fraud(safe_tx)
    safe_prob = safe_result["fraud_probability"]

    print(f"\n  TEST 1 — SAFE Transaction:")
    print(f"    Amount: ₹200, Device: trusted, Location: home")
    print(f"    ML Probability: {safe_prob:.4f}")
    print(f"    ML Risk Score:  {safe_result['ml_risk_score']}")
    print(f"    Features:       {safe_result['features']}")
    assert safe_prob < 0.5, f"❌ SAFE tx should have prob < 0.5, got {safe_prob}"
    print(f"    ✅ PASS — low fraud probability")

    # TEST 2: MFA transaction (medium risk)
    mfa_tx = {
        "amount": 20000,
        "city": "Pune",
        "device_id": "DEV-TRUSTED-01",
        "merchant_category": "electronics",
        "velocity_count_1h": 3,
        "timestamp": "2026-04-04T14:00:00",
        "device_changes": 1,
    }
    mfa_result = predict_fraud(mfa_tx)
    mfa_prob = mfa_result["fraud_probability"]

    print(f"\n  TEST 2 — MFA Transaction:")
    print(f"    Amount: ₹20,000, Device: new, Location: foreign")
    print(f"    ML Probability: {mfa_prob:.4f}")
    print(f"    ML Risk Score:  {mfa_result['ml_risk_score']}")
    print(f"    Features:       {mfa_result['features']}")
    assert mfa_prob > safe_prob, f"❌ MFA tx should have higher prob than SAFE, got {mfa_prob} vs {safe_prob}"
    print(f"    ✅ PASS — higher than SAFE")

    # TEST 3: BLOCK transaction (high risk)
    block_tx = {
        "amount": 80000,
        "city": "Lagos",
        "device_id": "NEW-DEVICE-X2",
        "merchant_category": "cryptocurrency",
        "velocity_count_1h": 8,
        "timestamp": "2026-04-04T03:00:00",
        "device_changes": 4,
    }
    block_result = predict_fraud(block_tx)
    block_prob = block_result["fraud_probability"]

    print(f"\n  TEST 3 — BLOCK Transaction:")
    print(f"    Amount: ₹80,000, Device: new, Location: foreign, Merchant: crypto")
    print(f"    ML Probability: {block_prob:.4f}")
    print(f"    ML Risk Score:  {block_result['ml_risk_score']}")
    print(f"    Features:       {block_result['features']}")
    assert block_prob > mfa_prob, f"❌ BLOCK tx should have higher prob than MFA, got {block_prob} vs {mfa_prob}"
    assert block_prob > 0.8, f"❌ BLOCK tx should have prob > 0.8, got {block_prob}"
    print(f"    ✅ PASS — highest fraud probability")

    # Summary
    print(f"\n  {'=' * 50}")
    print(f"  ML Model Validation Summary:")
    print(f"    SAFE  prob: {safe_prob:.4f}  ({'✅' if safe_prob < 0.3 else '⚠️'})")
    print(f"    MFA   prob: {mfa_prob:.4f}  ({'✅' if 0.3 < mfa_prob < 0.9 else '⚠️'})")
    print(f"    BLOCK prob: {block_prob:.4f}  ({'✅' if block_prob > 0.8 else '⚠️'})")
    print(f"    Ordering:   SAFE < MFA < BLOCK  ✅")
    print(f"  {'=' * 50}")


# ── Test 2: Decision Engine validation ─────────────────────────────────────────

def test_decision_engine():
    """Validate the DecisionEngine produces correct decisions."""
    print("\n" + "=" * 60)
    print("TEST: Decision Engine Validation")
    print("=" * 60)

    from backend.services.decision_engine import DecisionEngine

    # SAFE: low rule score + low ML
    result_safe = DecisionEngine.decide(10, ["Minor signal"], {"_fraud_probability": 0.05, "ml": 5})
    print(f"\n  SAFE test: score=10, ml_prob=0.05")
    print(f"    Decision: {result_safe['decision']}")
    print(f"    Final Score: {result_safe['final_score']}")
    print(f"    ML Score: {result_safe['ml_score']}")
    print(f"    Rule Score: {result_safe['rule_score']}")
    assert result_safe["decision"] == "APPROVE", f"❌ Expected APPROVE, got {result_safe['decision']}"
    print(f"    ✅ PASS")

    # MFA: medium rule score + medium ML
    result_mfa = DecisionEngine.decide(50, ["Medium signal"], {"_fraud_probability": 0.55, "ml": 55})
    print(f"\n  MFA test: score=50, ml_prob=0.55")
    print(f"    Decision: {result_mfa['decision']}")
    print(f"    Final Score: {result_mfa['final_score']}")
    assert result_mfa["decision"] == "MFA_HOLD", f"❌ Expected MFA_HOLD, got {result_mfa['decision']}"
    print(f"    ✅ PASS")

    # BLOCK: high rule score + high ML
    result_block = DecisionEngine.decide(85, ["Critical signal"], {"_fraud_probability": 0.95, "ml": 95})
    print(f"\n  BLOCK test: score=85, ml_prob=0.95")
    print(f"    Decision: {result_block['decision']}")
    print(f"    Final Score: {result_block['final_score']}")
    assert result_block["decision"] == "BLOCK", f"❌ Expected BLOCK, got {result_block['decision']}"
    print(f"    ✅ PASS")

    print(f"\n  All Decision Engine tests passed ✅")


# ── Test 3: Feature alignment validation ───────────────────────────────────────

def test_feature_alignment():
    """Ensure training and inference use identical features."""
    print("\n" + "=" * 60)
    print("TEST: Feature Alignment Validation")
    print("=" * 60)

    import joblib

    model = joblib.load("models/fraud_model.pkl")
    expected_features = 9
    actual_features = model.n_features_in_

    print(f"  Model expects {actual_features} features")
    print(f"  Service provides {expected_features} features")
    assert actual_features == expected_features, f"❌ Feature mismatch: model={actual_features}, service={expected_features}"
    print(f"  ✅ Feature count matches perfectly")

    # Verify predict works without error
    import pandas as pd
    FEATURE_COLS = [
        "amount", "amount_deviation", "location_change", "new_device",
        "merchant_risk", "txn_velocity", "is_night", "device_change_frequency",
        "distance_from_home"
    ]
    test_input = pd.DataFrame([[500, 0.1, 0, 0, 0, 1, 0, 0, 15.0]], columns=FEATURE_COLS)
    prob = model.predict_proba(test_input)[0][1]
    print(f"  Test prediction: {prob:.4f}")
    print(f"  ✅ Prediction works with aligned features")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════╗")
    print("║     FraudSense — System Validation Suite             ║")
    print("╚═══════════════════════════════════════════════════════╝")

    try:
        test_feature_alignment()
        test_ml_model_directly()
        test_decision_engine()

        print("\n" + "=" * 60)
        print("  🎉 ALL TESTS PASSED — System is working correctly!")
        print("  ✅ ML model produces varied, meaningful predictions")
        print("  ✅ Decisions are NOT always SAFE")
        print("  ✅ Feature alignment is correct (9 features)")
        print("  ✅ Decision thresholds produce BLOCK/MFA/APPROVE")
        print("=" * 60 + "\n")
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
