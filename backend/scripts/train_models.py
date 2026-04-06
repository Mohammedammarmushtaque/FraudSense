"""
train_models.py
FraudSense — ML Model Training Pipeline

Generates 2000 realistic synthetic transactions and trains a RandomForest
classifier on 8 features with deterministic fraud labeling logic.

Features (STRICT ORDER):
  [amount, amount_deviation, location_change, new_device, merchant_risk,
   txn_velocity, is_night, device_change_frequency]
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data.csv"
MODEL_PATH = "models/fraud_model.pkl"

FEATURE_COLS = [
    "amount", "amount_deviation", "location_change", "new_device",
    "merchant_risk", "txn_velocity", "is_night", "device_change_frequency",
    "distance_from_home"
]


def generate_synthetic_data(num_rows=2000):
    """
    Generate realistic synthetic fraud dataset with deterministic labeling.

    Fraud label = 1 if ANY of:
      - amount > 50000
      - location_change == 1
      - new_device == 1
      - txn_velocity > 6
      - merchant_risk == 1

    This matches the rule engine's logic so ML and rules are correlated
    but ML learns relative importance through tree splits.
    """
    print(f"Generating {num_rows} synthetic transactions...")
    np.random.seed(42)

    data = []
    for _ in range(num_rows):
        # Decide fraud probability bucket first (30% fraud, 70% legit)
        is_fraud_scenario = np.random.rand() < 0.30

        if not is_fraud_scenario:
            # ── LEGITIMATE TRANSACTION ────────────────────────────────────
            amount = np.random.uniform(100, 15000)
            amount_deviation = amount / np.random.uniform(3000, 20000)
            distance_from_home = np.random.uniform(0, 400) if np.random.rand() < 0.92 else np.random.uniform(501, 1500)
            location_change = 1 if distance_from_home > 500 else 0
            new_device = 0 if np.random.rand() < 0.95 else 1
            merchant_risk = 0 if np.random.rand() < 0.95 else 1
            txn_velocity = np.random.randint(1, 4)
            is_night = 0 if np.random.rand() < 0.88 else 1
            device_change_frequency = 0 if np.random.rand() < 0.85 else np.random.randint(1, 3)
        else:
            # ── FRAUDULENT TRANSACTION ────────────────────────────────────
            # Mix of strong and subtle fraud signals
            fraud_type = np.random.choice(["high_amount", "location", "device", "velocity", "mixed"])

            if fraud_type == "high_amount":
                amount = np.random.uniform(50001, 100000)
                distance_from_home = np.random.uniform(0, 800)
                new_device = int(np.random.rand() < 0.3)
                merchant_risk = int(np.random.rand() < 0.5)
                txn_velocity = np.random.randint(1, 8)
            elif fraud_type == "location":
                amount = np.random.uniform(5000, 60000)
                distance_from_home = np.random.uniform(501, 5000)
                new_device = int(np.random.rand() < 0.6)
                merchant_risk = int(np.random.rand() < 0.4)
                txn_velocity = np.random.randint(2, 7)
            elif fraud_type == "device":
                amount = np.random.uniform(2000, 80000)
                distance_from_home = np.random.uniform(0, 1000)
                new_device = 1
                merchant_risk = int(np.random.rand() < 0.5)
                txn_velocity = np.random.randint(1, 6)
            elif fraud_type == "velocity":
                amount = np.random.uniform(1000, 50000)
                distance_from_home = np.random.uniform(0, 600)
                new_device = int(np.random.rand() < 0.4)
                merchant_risk = int(np.random.rand() < 0.4)
                txn_velocity = np.random.randint(7, 10)
            else:  # mixed — multiple signals
                amount = np.random.uniform(30000, 100000)
                distance_from_home = np.random.uniform(501, 5000)
                new_device = 1
                merchant_risk = 1
                txn_velocity = np.random.randint(5, 10)

            location_change = 1 if distance_from_home > 500 else 0
            amount_deviation = amount / np.random.uniform(1000, 10000)
            is_night = int(np.random.rand() < 0.55)
            device_change_frequency = np.random.randint(1, 4)

        # ── DETERMINISTIC LABEL ───────────────────────────────────────────
        label = 1 if (
            amount > 50000 or
            location_change == 1 or
            new_device == 1 or
            txn_velocity > 6 or
            merchant_risk == 1
        ) else 0

        data.append([
            round(amount, 2),
            round(amount_deviation, 4),
            location_change,
            new_device,
            merchant_risk,
            txn_velocity,
            is_night,
            device_change_frequency,
            round(distance_from_home, 2),
            label
        ])

    df = pd.DataFrame(data, columns=FEATURE_COLS + ["label"])
    df.to_csv(DATA_PATH, index=False)

    fraud_count = df["label"].sum()
    print(f"Dataset saved to {DATA_PATH}")
    print(f"  Total: {len(df)} | Fraud: {fraud_count} ({fraud_count/len(df)*100:.1f}%) | Legit: {len(df)-fraud_count}")

    return df


def main():
    # ── 1. Generate or load data ──────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        df = generate_synthetic_data(2000)
    else:
        print(f"Loading existing data from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH)

        # Validate columns exist
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing or "label" not in df.columns:
            print(f"Schema mismatch — regenerating dataset (missing: {missing})")
            df = generate_synthetic_data(2000)

    # ── 2. Feature / label split ──────────────────────────────────────────────
    X = df[FEATURE_COLS]
    y = df["label"]

    assert X.shape[1] == 9, f"Expected 9 features, got {X.shape[1]}"
    print(f"Feature matrix: {X.shape}, Label distribution: {dict(y.value_counts())}")

    # ── 3. Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 4. Train RandomForestClassifier ───────────────────────────────────────
    print("Training RandomForestClassifier (200 trees, max_depth=12)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # Verify feature count matches
    assert model.n_features_in_ == 9, f"Model trained on {model.n_features_in_} features, expected 9"

    # ── 6. Save model ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Features trained on: {FEATURE_COLS}")

    # ── 7. Quick smoke test ───────────────────────────────────────────────────
    test_safe = pd.DataFrame([[200, 0.04, 0, 0, 0, 1, 0, 0, 15.0]], columns=FEATURE_COLS)
    test_block = pd.DataFrame([[80000, 8.0, 1, 1, 1, 8, 1, 4, 850.0]], columns=FEATURE_COLS)

    prob_safe = model.predict_proba(test_safe)[0][1]
    prob_block = model.predict_proba(test_block)[0][1]
    print(f"\n🔬 Smoke test:")
    print(f"  SAFE  tx (amount=200):   fraud_prob = {prob_safe:.4f}")
    print(f"  BLOCK tx (amount=80000): fraud_prob = {prob_block:.4f}")

    if prob_block > prob_safe:
        print("  ✅ Model correctly ranks BLOCK > SAFE")
    else:
        print("  ⚠️  Model ranking may be off — check training data")


if __name__ == "__main__":
    main()