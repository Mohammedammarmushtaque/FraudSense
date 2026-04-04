"""Quick focused test — run with: python quick_test.py"""
import sys, os
sys.path.insert(0, ".")

from backend.services.ml_service import predict_fraud
from backend.services.decision_engine import DecisionEngine

print("=" * 60)
print("FOCUSED FRAUD SYSTEM VALIDATION")
print("=" * 60)

# SAFE
r1 = predict_fraud({"amount": 200, "city": "Mumbai", "device_id": "DEV-TRUSTED-01",
    "merchant_category": "groceries", "velocity_count_1h": 1,
    "timestamp": "2026-04-04T10:00:00", "device_changes": 0})
p1 = r1["fraud_probability"]
print("SAFE  -> prob=%.4f  score=%d" % (p1, r1["ml_risk_score"]))

# MFA
r2 = predict_fraud({"amount": 20000, "city": "Pune", "device_id": "DEV-TRUSTED-01",
    "merchant_category": "electronics", "velocity_count_1h": 3,
    "timestamp": "2026-04-04T14:00:00", "device_changes": 1})
p2 = r2["fraud_probability"]
print("MFA   -> prob=%.4f  score=%d" % (p2, r2["ml_risk_score"]))

# BLOCK
r3 = predict_fraud({"amount": 80000, "city": "Lagos", "device_id": "NEW-X2",
    "merchant_category": "cryptocurrency", "velocity_count_1h": 8,
    "timestamp": "2026-04-04T03:00:00", "device_changes": 4})
p3 = r3["fraud_probability"]
print("BLOCK -> prob=%.4f  score=%d" % (p3, r3["ml_risk_score"]))

print()

# Decision engine tests
d1 = DecisionEngine.decide(10, [], {"_fraud_probability": p1})
print("SAFE  -> decision=%s  final=%.4f" % (d1["decision"], d1["final_score"]))

d2 = DecisionEngine.decide(50, ["foreign location"], {"_fraud_probability": p2})
print("MFA   -> decision=%s  final=%.4f" % (d2["decision"], d2["final_score"]))

d3 = DecisionEngine.decide(85, ["crypto", "new device"], {"_fraud_probability": p3})
print("BLOCK -> decision=%s  final=%.4f" % (d3["decision"], d3["final_score"]))

print()
assert d1["decision"] == "APPROVE", "FAIL: safe tx not APPROVE"
assert d2["decision"] == "MFA_HOLD", "FAIL: mfa tx not MFA_HOLD"
assert d3["decision"] == "BLOCK", "FAIL: block tx not BLOCK"
assert p1 < p2 < p3, "FAIL: probabilities not in correct order"
print("ALL 3 TESTS PASSED - System produces varied decisions!")
