"""
backend/services/ml_model.py
FraudSense — RandomForest Model Wrapper

Provides a FraudModel class and score_transaction() compatibility wrapper.
Uses the SAME 8-feature contract as ml_service.py and train_models.py.
"""

import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "models/fraud_model.pkl"

FEATURE_COLS = [
    "amount", "amount_deviation", "location_change", "new_device",
    "merchant_risk", "txn_velocity", "is_night", "device_change_frequency"
]

RISKY_MERCHANTS = {
    "crypto", "unknown", "high-risk", "cryptocurrency",
    "wire_transfer", "gambling", "casino", "forex"
}


class FraudModel:
    """
    RandomForest-based Fraud Prediction Model.
    Trained on 8 features to perfectly align training and prediction.
    """
    def __init__(self):
        self.model = None
        self._is_trained = False

    def train(self):
        """Generates synthetic samples and trains the model."""
        print("Training RandomForest fraud model...")
        np.random.seed(42)
        n_samples = 2000

        amounts = np.random.uniform(100, 100000, n_samples)
        amount_deviations = amounts / np.random.uniform(1000, 20000, n_samples)
        location_changes = np.random.randint(0, 2, n_samples)
        new_devices = np.random.randint(0, 2, n_samples)
        merchant_risks = np.random.randint(0, 2, n_samples)
        txn_velocities = np.random.randint(1, 11, n_samples)
        is_nights = np.random.randint(0, 2, n_samples)
        device_change_frequencies = np.random.randint(0, 6, n_samples)

        X = np.column_stack([
            amounts, amount_deviations, location_changes, new_devices,
            merchant_risks, txn_velocities, is_nights, device_change_frequencies
        ])

        # Deterministic fraud labeling
        y = (
            (amounts > 50000) |
            (location_changes == 1) |
            (new_devices == 1) |
            (txn_velocities > 6) |
            (merchant_risks == 1)
        ).astype(int)

        print(f"Dataset shape: {X.shape}, Fraud ratio: {y.mean():.2%}")

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            min_samples_split=5, random_state=42, n_jobs=-1
        )
        self.model.fit(X, y)
        self._is_trained = True

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    def predict_proba(self, X):
        """Returns probability of fraud [0, 1]"""
        if not self._is_trained:
            if os.path.exists(MODEL_PATH):
                print("Loading model from:", MODEL_PATH)
                self.model = joblib.load(MODEL_PATH)
                self._is_trained = True
            else:
                self.train()

        probs = self.model.predict_proba(X)
        return probs[:, 1]


def train_lightweight_model():
    """Compatibility wrapper for existing calls."""
    m = FraudModel()
    m.train()


def score_transaction(tx_data):
    """
    Compatibility wrapper for scoring engine.
    Extracts the identical 8 features used during training.
    """
    m = FraudModel()

    amount = float(tx_data.get('amount', 0))
    amount_deviation = amount / max(float(tx_data.get("avg_amount", 5000)), 1.0)

    home_location = tx_data.get("home_location", "Mumbai")
    location = tx_data.get("city", tx_data.get("location", "Mumbai"))
    location_change = 1 if location.lower().strip() != home_location.lower().strip() else 0

    trusted_device = tx_data.get("trusted_device", "DEV-TRUSTED-01")
    device = tx_data.get("device_id", tx_data.get("device", "DEV-TRUSTED-01"))
    new_device = 1 if device != trusted_device else 0

    merchant = str(tx_data.get("merchant_category", tx_data.get("merchant", "unknown"))).lower()
    merchant_risk = 1 if merchant in RISKY_MERCHANTS else 0

    txn_velocity = int(tx_data.get("txn_last_5min", tx_data.get("velocity_count_1h", 1)))

    hour = 12
    if "timestamp" in tx_data and tx_data["timestamp"]:
        try:
            from datetime import datetime
            ts = tx_data["timestamp"]
            if isinstance(ts, str):
                hour = datetime.fromisoformat(ts).hour
            elif hasattr(ts, "hour"):
                hour = ts.hour
        except Exception:
            pass
    hour = int(tx_data.get("hour", hour))
    is_night = 1 if hour < 6 or hour > 22 else 0

    device_change_frequency = int(tx_data.get("device_changes", 0))

    X = np.array([[
        amount, amount_deviation, location_change, new_device,
        merchant_risk, txn_velocity, is_night, device_change_frequency
    ]])

    features_list = X[0].tolist()
    print("Input features:", features_list)

    try:
        prob = m.predict_proba(X)[0]
    except Exception as e:
        raise RuntimeError(f"ML prediction failed: {str(e)}")

    return {
        "ml_anomaly_prob": round(float(prob), 4),
        "raw_score": float(prob),
        "reason": f"Fraud probability {prob:.2f} (RandomForest)"
    }
