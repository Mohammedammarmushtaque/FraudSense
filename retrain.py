# retrain.py
# Adaptive retraining and drift monitoring for FraudSense
# Runs as a background daemon thread, checking every 30 seconds
# for new confirmed fraud cases and model drift signals

import threading
import time
import sqlite3
import json
import shutil
import datetime

import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
)
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. SMOTE disabled.")


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Compares the most-recent 100 transactions against the prior 100 to
    surface two distinct fraud-evasion signals:

      1. BOUNDARY_PROBING  – the MFA-hold rate has spiked, suggesting
                             fraudsters are deliberately sitting just
                             below the decision threshold.

      2. EVASION_IMPROVING – confirmed fraud cases are accumulating while
                             the average risk score is falling, meaning the
                             model is being fooled more successfully over
                             time.

    Every detected signal is persisted to the drift_alerts table so the
    dashboard can surface it without re-querying transaction history.
    """

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.window_size = 100          # rows per comparison window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check_score_drift(self) -> list[dict]:
        """
        Run both drift checks and return a (possibly empty) list of
        signal dictionaries, each with keys:
            alert_type, message, recommendation
        """
        recent = self._fetch_window(offset=0)
        prior  = self._fetch_window(offset=self.window_size)

        # Guard: not enough historical data yet
        if recent is None or prior is None:
            return []

        drift_signals: list[dict] = []

        # ---- Signal 1: MFA rate spike (boundary probing) ---------------
        recent_mfa = recent["mfa_rate"] or 0.0
        prior_mfa  = prior["mfa_rate"]  or 0.0

        if prior_mfa > 0 and recent_mfa > prior_mfa * 1.5:
            drift_signals.append({
                "alert_type": "BOUNDARY_PROBING",
                "message": (
                    f"MFA rate rose from {prior_mfa:.1f}% to "
                    f"{recent_mfa:.1f}% — fraudsters may be probing "
                    "the verification threshold"
                ),
                "recommendation": (
                    "Consider tightening the MFA threshold by 5 points"
                ),
            })

        # ---- Signal 2: Scores falling while fraud confirmed (evasion) --
        confirmed_recent = self._count_recent_confirmed_fraud(hours=1)
        recent_avg = recent["avg_score"] or 0.0
        prior_avg  = prior["avg_score"]  or 0.0

        if confirmed_recent >= 3 and prior_avg > 0 and recent_avg < prior_avg * 0.85:
            drift_signals.append({
                "alert_type": "EVASION_IMPROVING",
                "message": (
                    "Confirmed fraud cases increasing while average risk "
                    "score is decreasing — fraud tactics are evolving "
                    "to evade the current model"
                ),
                "recommendation": "Trigger an immediate model retrain",
            })

        # ---- Persist every signal found --------------------------------
        self._store_signals(drift_signals)

        return drift_signals

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_window(self, offset: int) -> sqlite3.Row | None:
        """
        Return aggregated stats for a window of self.window_size rows,
        starting at *offset* rows from the most-recent transaction.
        Returns None when the slice is empty (too little data).
        """
        row = self.db.execute(
            """
            SELECT
                AVG(risk_score) AS avg_score,
                SUM(CASE WHEN decision = 'MFA_HOLD' THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*) AS mfa_rate
            FROM (
                SELECT risk_score, decision
                FROM   transactions
                ORDER  BY timestamp DESC
                LIMIT  ?
                OFFSET ?
            )
            """,
            (self.window_size, offset),
        ).fetchone()

        # fetchone() always returns a Row; check that we actually got data
        if row is None or row["avg_score"] is None:
            return None
        return row

    def _count_recent_confirmed_fraud(self, hours: int = 1) -> int:
        """
        Count how many confirmed-fraud entries in retraining_buffer are
        linked to transactions from the last *hours* hours.
        """
        row = self.db.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM   transactions t
            JOIN   retraining_buffer rb ON t.tx_id = rb.tx_id
            WHERE  t.timestamp > datetime('now', ?)
            """,
            (f"-{hours} hour",),
        ).fetchone()
        return row["cnt"] if row else 0

    def _store_signals(self, signals: list[dict]) -> None:
        """Persist drift signals and commit in one batch."""
        now = datetime.datetime.now().isoformat()
        for signal in signals:
            self.db.execute(
                """
                INSERT INTO drift_alerts
                    (alert_type, message, recommendation, detected_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    signal["alert_type"],
                    signal["message"],
                    signal["recommendation"],
                    now,
                ),
            )
        if signals:
            self.db.commit()


# ---------------------------------------------------------------------------
# RetrainWorker
# ---------------------------------------------------------------------------

class RetrainWorker(threading.Thread):
    """
    Background daemon that wakes every 30 seconds and:

      1. Counts rows in retraining_buffer (confirmed fraud labels from
         analysts / charge-back feeds).
      2. If >= 5 rows exist, retrains the XGBoost model on the union of
         the original PaySim dataset and the buffered cases (SMOTE-
         balanced, StandardScaler-normalised).
      3. Saves versioned artefacts and atomically replaces the live
         model files read by app.py.
      4. Runs DriftMonitor and logs any signals it finds.

    Thread is set as a daemon so it exits automatically when the main
    Flask/Gunicorn process terminates.
    """

    # Minimum confirmed-fraud records needed to trigger a retrain
    RETRAIN_THRESHOLD: int = 5
    # Seconds between each check loop
    POLL_INTERVAL: int = 30

    def __init__(self, db_path: str):
        super().__init__(daemon=True)
        self.db_path = db_path
        self.running  = True           # set to False to stop gracefully

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("RetrainWorker started -- polling every "
              f"{self.POLL_INTERVAL}s")

        while self.running:
            time.sleep(self.POLL_INTERVAL)

            # Each iteration gets its own short-lived connection so we
            # never hold a lock across the sleep window.
            db = self._open_db()
            try:
                count = self._buffer_count(db)

                if count >= self.RETRAIN_THRESHOLD:
                    print(f"Retraining triggered "
                          f"({count} confirmed fraud cases in buffer)")
                    self._retrain_model(db)

                    # Drift check runs after retrain so the new metrics
                    # are already in model_metrics when we read them.
                    monitor = DriftMonitor(db)
                    signals = monitor.check_score_drift()
                    if signals:
                        print(f"Drift detected: "
                              f"{len(signals)} signal(s)")
                        for s in signals:
                            print(f"   [{s['alert_type']}] {s['message']}")
                else:
                    print(f"Buffer at {count}/{self.RETRAIN_THRESHOLD} "
                          "— no retrain needed")

            except Exception as exc:            # never let the thread die silently
                print(f"RetrainWorker error: {exc}")

            finally:
                db.close()

    # ------------------------------------------------------------------
    # Core retrain logic
    # ------------------------------------------------------------------

    def _retrain_model(self, db: sqlite3.Connection) -> None:
        """
        Full retrain pipeline:
            load data → feature engineering → SMOTE → scale →
            XGBoost fit → evaluate → save → atomic deploy → log metrics
        """

        # ---- 1. Load base dataset --------------------------------------
        try:
            original_data = pd.read_csv("data/paysim_processed.csv")
        except FileNotFoundError:
            print("❌ data/paysim_processed.csv not found — aborting retrain")
            return

        # ---- 2. Load confirmed fraud from buffer -----------------------
        rows = db.execute(
            "SELECT feature_vector, is_fraud FROM retraining_buffer"
        ).fetchall()

        if not rows:
            print("⚠️  retraining_buffer is empty — skipping")
            return

        try:
            buffer_records = [
                {**json.loads(row["feature_vector"]),
                 "isFraud": int(row["is_fraud"])}
                for row in rows
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"❌ Could not parse buffer rows: {exc}")
            return

        buffer_df = pd.DataFrame(buffer_records)

        # ---- 3. Combine datasets ---------------------------------------
        combined = pd.concat(
            [original_data, buffer_df],
            ignore_index=True,
        )

        # ---- 4. Feature / label split ----------------------------------
        feature_cols = [
            "amount",
            "balance_diff_orig",
            "balance_diff_dest",
            "hour_of_day",
            "tx_type_encoded",
            "amount_to_balance_ratio",
        ]

        missing = [c for c in feature_cols if c not in combined.columns]
        if missing:
            print(f"❌ Missing feature columns: {missing} — aborting retrain")
            return

        X = combined[feature_cols].copy()
        y = combined["isFraud"].copy()

        # Drop any rows with NaNs introduced by the buffer merge
        valid_mask = X.notna().all(axis=1) & y.notna()
        X, y = X[valid_mask], y[valid_mask]

        # ---- 5. Train / test split ------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,        # preserve class ratio in both splits
        )

        # ---- 6. SMOTE oversampling on training set --------------------
        if _SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            try:
                X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            except ValueError as exc:
                # SMOTE requires at least 2 fraud examples; degrade gracefully
                print(f"⚠️  SMOTE failed ({exc}) — training on imbalanced data")
                X_train_bal, y_train_bal = X_train, y_train
        else:
            print("⚠️  SMOTE not available — training on imbalanced data")
            X_train_bal, y_train_bal = X_train, y_train

        # ---- 7. Feature scaling ----------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_bal)
        X_test_scaled  = scaler.transform(X_test)

        # ---- 8. Fit XGBoost --------------------------------------------
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="auprc",
            random_state=42,
            verbosity=0,       # suppress XGBoost console spam
        )
        model.fit(X_train_scaled, y_train_bal)

        # ---- 9. Evaluate -----------------------------------------------
        y_pred       = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", zero_division=0
        )
        auprc = average_precision_score(y_test, y_pred_proba)

        # ---- 10. Determine next version number -------------------------
        row = db.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS next_ver "
            "FROM model_metrics"
        ).fetchone()
        version = row["next_ver"]

        # ---- 11. Save versioned artefacts ------------------------------
        versioned_model  = f"models/xgboost_model_v{version}.pkl"
        versioned_scaler = f"models/feature_scaler_v{version}.pkl"

        joblib.dump(model,  versioned_model)
        joblib.dump(scaler, versioned_scaler)

        # ---- 12. Atomic deploy (copy → overwrite live files) -----------
        shutil.copy(versioned_model,  "models/xgboost_model.pkl")
        shutil.copy(versioned_scaler, "models/feature_scaler.pkl")

        # ---- 13. Log metrics to DB -------------------------------------
        db.execute(
            """
            INSERT INTO model_metrics
                (version, precision, recall, f1_score, auprc, trained_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                float(precision),
                float(recall),
                float(f1),
                float(auprc),
                datetime.datetime.now().isoformat(),
            ),
        )
        db.commit()

        print(
            f"✅ Model v{version} deployed | "
            f"Precision: {precision:.3f}  "
            f"Recall: {recall:.3f}  "
            f"AUPRC: {auprc:.3f}"
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _open_db(self) -> sqlite3.Connection:
        """Open a Row-factory–enabled connection to the project DB."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _buffer_count(self, db: sqlite3.Connection) -> int:
        """Return the current number of rows in retraining_buffer."""
        row = db.execute(
            "SELECT COUNT(*) AS cnt FROM retraining_buffer"
        ).fetchone()
        return row["cnt"] if row else 0

    def stop(self) -> None:
        """Signal the worker to exit after its next sleep cycle."""
        self.running = False


# ---------------------------------------------------------------------------
# Module-level convenience: start the worker from any entry point
# ---------------------------------------------------------------------------

def start_retrain_worker(db_path: str = "fraudsense.db") -> RetrainWorker:
    """
    Instantiate and start a RetrainWorker, then return the thread
    object so callers can call .stop() later if needed.

    Usage (e.g. inside app.py create_app):
        from retrain import start_retrain_worker
        start_retrain_worker("fraudsense.db")
    """
    worker = RetrainWorker(db_path)
    worker.start()
    return worker


# ---------------------------------------------------------------------------
# Direct execution entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # When run directly (python retrain.py) keep the main thread alive
    # so the daemon worker has a parent process to attach to.
    worker = start_retrain_worker("fraudsense.db")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping RetrainWorker...")
        worker.stop()
        worker.join(timeout=10)
        print("👋 Shutdown complete")