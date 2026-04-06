"""
backend/services/ml_service.py
FraudSense — ML Prediction Service

Loads the RandomForest model trained by train_models.py and provides
predict_fraud() and explain_prediction() for the risk pipeline.

STRICT 8-feature contract:
  [amount, amount_deviation, location_change, new_device, merchant_risk,
   txn_velocity, is_night, device_change_frequency]
"""

import joblib
import os
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "models/fraud_model.pkl"

import math

FEATURE_COLS = [
    "amount", "amount_deviation", "location_change", "new_device",
    "merchant_risk", "txn_velocity", "is_night", "device_change_frequency",
    "distance_from_home"
]

LOCATION_COORDS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567)
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0 # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ── Model singleton ───────────────────────────────────────────────────────────
model = None


def train_model():
    """Run the training pipeline to create a fresh model file."""
    print("⚙ Training model via backend/scripts/train_models.py ...")
    script_path = os.path.join("backend", "scripts", "train_models.py")
    subprocess.run([sys.executable, script_path], check=True)
    print("✅ Training complete.")


def _load_model():
    """Load the model from disk, training first if necessary."""
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠  Model not found at {MODEL_PATH} — training now...")
        train_model()

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file {MODEL_PATH} not found even after training. "
            "Check train_models.py for errors."
        )

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # Validate feature count
    expected = len(FEATURE_COLS)
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and actual != expected:
        raise RuntimeError(
            f"Model expects {actual} features but service provides {expected}. "
            "Delete models/fraud_model.pkl and restart to retrain."
        )

    logger.info(f"Loaded ML model from {MODEL_PATH} ({actual} features)")


# ── Eager load at import time ──────────────────────────────────────────────────
_load_model()


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features(txn: dict) -> dict:
    """
    Extract the 8 features in STRICT order matching the training pipeline.

    Returns a dict keyed by feature name with numeric values.
    """
    amount = float(txn.get("amount", 0))
    amount_deviation = amount / max(float(txn.get("avg_amount", 5000)), 1.0)

    # Location anomaly using Haversine
    USER_BASE_LOCATION = "Mumbai"
    home_location = txn.get("home_location", USER_BASE_LOCATION)
    location = txn.get("city", txn.get("location", "Mumbai"))
    
    distance = 0.0
    if home_location in LOCATION_COORDS and location in LOCATION_COORDS:
        h_lat, h_lon = LOCATION_COORDS[home_location]
        l_lat, l_lon = LOCATION_COORDS[location]
        distance = haversine(h_lat, h_lon, l_lat, l_lon)
    else:
        # Fallback if city not in map
        distance = 0.0 if location.lower().strip() == home_location.lower().strip() else 1000.0

    print(f"Location: {location}")
    print(f"Distance from home: {distance:.2f} km")

    location_change = 1 if distance > 500 else 0
    distance_from_home = float(distance)

    # New device
    trusted_device = txn.get("trusted_device", "DEV-TRUSTED-01")
    device = txn.get("device_id", txn.get("device", "DEV-TRUSTED-01"))
    new_device = 1 if device != trusted_device else 0

    # Merchant risk
    merchant = str(txn.get("merchant_category", txn.get("merchant", "unknown"))).lower()
    RISKY_MERCHANTS = {"crypto", "unknown", "high-risk", "cryptocurrency", "wire_transfer", "gambling", "casino", "forex"}
    merchant_risk = 1 if merchant in RISKY_MERCHANTS else 0

    # Transaction velocity
    txn_velocity = int(txn.get("txn_last_5min", txn.get("velocity_count_1h", 1)))

    # Night flag
    hour = 12
    if "timestamp" in txn and txn["timestamp"]:
        try:
            from datetime import datetime
            ts = txn["timestamp"]
            if isinstance(ts, str):
                hour = datetime.fromisoformat(ts).hour
            elif hasattr(ts, "hour"):
                hour = ts.hour
        except Exception:
            pass
    hour = int(txn.get("hour", hour))
    is_night = 1 if hour < 6 or hour > 22 else 0

    # Device change frequency
    device_change_frequency = int(txn.get("device_changes", 0))

    return {
        "amount": amount,
        "amount_deviation": round(amount_deviation, 4),
        "location_change": location_change,
        "new_device": new_device,
        "merchant_risk": merchant_risk,
        "txn_velocity": txn_velocity,
        "is_night": is_night,
        "device_change_frequency": device_change_frequency,
        "distance_from_home": distance_from_home,
    }


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_fraud(txn: dict) -> dict:
    """
    Run ML prediction on a transaction dict.

    Returns:
        {
            "fraud_probability": float (0-1),
            "ml_risk_score": int (0-100),
            "features": dict
        }

    Raises RuntimeError on any failure — NEVER silently returns SAFE.
    """
    global model

    if model is None:
        raise RuntimeError("ML model is not loaded. Cannot predict — refusing to default to SAFE.")

    features_dict = extract_features(txn)

    # Build feature vector in STRICT column order
    features_list = [features_dict[col] for col in FEATURE_COLS]

    # Validate feature count
    assert len(features_list) == model.n_features_in_, (
        f"Feature length mismatch: expected {model.n_features_in_}, got {len(features_list)}"
    )

    try:
        import pandas as pd
        X = pd.DataFrame([features_list], columns=FEATURE_COLS)
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        raise RuntimeError(f"ML prediction failed: {str(e)}")

    ml_score = int(prob * 100)

    # ── Debug visibility ──────────────────────────────────────────────────
    print("  [ML] Input:  %s" % features_list)
    print("  [ML] Prob:   %.4f  ->  score %d/100" % (prob, ml_score))

    return {
        "fraud_probability": float(prob),
        "ml_risk_score": ml_score,
        "features": features_dict,
    }


# ── Explainability ─────────────────────────────────────────────────────────────

def explain_prediction(features: dict) -> list:
    """
    Generate human-readable reasons based on feature values.
    These are injected into the API response for the dashboard.
    """
    reasons = []

    amount = features.get("amount", 0)
    if amount > 50000:
        reasons.append("High transaction amount (>₹50,000)")
    elif amount > 20000:
        reasons.append("Elevated transaction amount (>₹20,000)")

    if features.get("location_change"):
        dist = features.get("distance_from_home", 0)
        reasons.append(f"Transaction from unusual location ({dist:.0f} km away)")

    if features.get("new_device"):
        reasons.append("New / unrecognized device")

    if features.get("txn_velocity", 0) > 5:
        reasons.append("High transaction frequency")
    elif features.get("txn_velocity", 0) > 3:
        reasons.append("Elevated transaction frequency")

    if features.get("merchant_risk"):
        reasons.append("High-risk merchant category")

    if features.get("is_night"):
        reasons.append("Transaction at unusual hour (night)")

    if features.get("device_change_frequency", 0) > 2:
        reasons.append("Frequent device changes detected")

    if features.get("amount_deviation", 0) > 5:
        reasons.append("Amount significantly deviates from user baseline")

    return reasons
