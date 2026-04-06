"""
backend/api/transaction.py
FraudSense – All API endpoints via FastAPI router
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
import asyncio
import datetime
from pathlib import Path

from backend.db.database import get_db_connection
from backend.services.profile_service import ProfileService
from backend.services.graph_service import GraphService
from backend.services.risk_engine import RiskEngine
from backend.services.decision_engine import DecisionEngine
from backend.services.behavioral import record_transaction
from backend.services.simulator import generate_batch, generate_single
from backend.services import llm_case_file

router = APIRouter()

# ============================================================================
# Constants
# ============================================================================

VALID_OTP = "1234"
CASE_CACHE_DIR = Path("case_cache")
CASE_FILES = {
    "ato":           "ato_case.json",
    "coordinated":   "coordinated_case.json",
    "mimicry":       "mimicry_case.json",
    "scam_romance":  "scam_romance.json",
    "stealth_probe": "stealth_probe_case.json",
}


# ============================================================================
# Pydantic Models
# ============================================================================

class TransactionRequest(BaseModel):
    tx_id: str
    user_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    device_id: str
    city: str
    device_timezone: str
    timestamp: str
    channel: str
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    tx_type: str


class MFARequest(BaseModel):
    tx_id: str
    otp_code: str
    purpose: Optional[str] = None


class AnalystConfirmRequest(BaseModel):
    tx_id: str
    is_fraud: bool
    analyst_id: str


class ChainResetRequest(BaseModel):
    user_id: str


class CoordinatedCheckRequest(BaseModel):
    merchant_id: str


class ScenarioRunRequest(BaseModel):
    scenario: str


# ============================================================================
# Helper: encode transaction type
# ============================================================================

def _encode_tx_type(tx_type: str) -> int:
    mapping = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2, "DEBIT": 3, "CASH_IN": 4}
    return mapping.get(tx_type.upper(), 1)


# ============================================================================
# Pre-built scenario definitions
# ============================================================================

SCENARIOS: dict[str, list[dict]] = {

    "ato": [
        {
            "tx_id": "ATO-001", "user_id": "user_ato_42",
            "amount": 4500, "merchant_id": "MERCH-CRYPTO-01",
            "merchant_category": "cryptocurrency", "device_id": "NEW-DEVICE-XYZ",
            "city": "Lagos", "device_timezone": "America/New_York",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "web", "oldbalanceOrg": 5000, "newbalanceOrig": 500,
            "oldbalanceDest": 0, "newbalanceDest": 4500, "tx_type": "TRANSFER",
        },
        {
            "tx_id": "ATO-002", "user_id": "user_ato_42",
            "amount": 499, "merchant_id": "MERCH-CRYPTO-01",
            "merchant_category": "cryptocurrency", "device_id": "NEW-DEVICE-XYZ",
            "city": "Lagos", "device_timezone": "America/New_York",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "mobile", "oldbalanceOrg": 500, "newbalanceOrig": 1,
            "oldbalanceDest": 0, "newbalanceDest": 499, "tx_type": "CASH_OUT",
        },
    ],

    "coordinated": [
        {
            "tx_id": f"COORD-{i:03d}", "user_id": f"user_coord_{i}",
            "amount": 300 + i * 10, "merchant_id": "MERCH-TARGET-99",
            "merchant_category": "electronics", "device_id": f"DEV-COORD-{i}",
            "city": "New York", "device_timezone": "America/New_York",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "web", "oldbalanceOrg": 1000, "newbalanceOrig": 700 - i * 10,
            "oldbalanceDest": 0, "newbalanceDest": 300 + i * 10,
            "tx_type": "PAYMENT",
        }
        for i in range(1, 7)
    ],

    "mimicry": [
        {
            "tx_id": "MIM-001", "user_id": "user_mimicry_77",
            "amount": 42.50, "merchant_id": "MERCH-GROCERY-07",
            "merchant_category": "groceries", "device_id": "DEV-KNOWN-77",
            "city": "Chicago", "device_timezone": "America/Chicago",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "mobile", "oldbalanceOrg": 2000, "newbalanceOrig": 1957.50,
            "oldbalanceDest": 0, "newbalanceDest": 42.50, "tx_type": "PAYMENT",
        },
        {
            "tx_id": "MIM-002", "user_id": "user_mimicry_77",
            "amount": 38.00, "merchant_id": "MERCH-COFFEE-03",
            "merchant_category": "restaurants", "device_id": "DEV-KNOWN-77",
            "city": "Chicago", "device_timezone": "America/Chicago",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "mobile", "oldbalanceOrg": 1957.50, "newbalanceOrig": 1919.50,
            "oldbalanceDest": 0, "newbalanceDest": 38.00, "tx_type": "PAYMENT",
        },
        {
            "tx_id": "MIM-003", "user_id": "user_mimicry_77",
            "amount": 9800.00, "merchant_id": "MERCH-WIRE-01",
            "merchant_category": "wire_transfer", "device_id": "DEV-KNOWN-77",
            "city": "Chicago", "device_timezone": "America/Chicago",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "web", "oldbalanceOrg": 1919.50, "newbalanceOrig": 0,
            "oldbalanceDest": 0, "newbalanceDest": 9800.00, "tx_type": "TRANSFER",
        },
    ],

    "scam": [
        {
            "tx_id": "SCAM-001", "user_id": "user_scam_99",
            "amount": 15000, "merchant_id": "MERCH-WIRE-88",
            "merchant_category": "wire_transfer", "device_id": "DEV-SCAM-99",
            "city": "Miami", "device_timezone": "America/New_York",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "web", "oldbalanceOrg": 20000, "newbalanceOrig": 5000,
            "oldbalanceDest": 0, "newbalanceDest": 15000, "tx_type": "TRANSFER",
        },
    ],

    "stealth_probe": [
        {
            "tx_id": f"PROBE-{i:03d}", "user_id": "user_probe_55",
            "amount": 34 + i,
            "merchant_id": "MERCH-DIGITAL-05",
            "merchant_category": "digital_goods", "device_id": "DEV-PROBE-55",
            "city": "Seattle", "device_timezone": "America/Los_Angeles",
            "timestamp": datetime.datetime.now().isoformat(),
            "channel": "api", "oldbalanceOrg": 5000 - i * 35,
            "newbalanceOrig": 5000 - (i + 1) * 35,
            "oldbalanceDest": 0, "newbalanceDest": 34 + i,
            "tx_type": "PAYMENT",
        }
        for i in range(1, 8)
    ],
}


# ============================================================================
# ENDPOINT 1 — POST /api/transaction
# ============================================================================

@router.post("/api/transaction")
async def process_transaction(req: TransactionRequest, background_tasks: BackgroundTasks):
    """
    Primary transaction evaluation endpoint — pre-transaction decisioning.
    Returns a structured decision BEFORE the transaction completes, allowing
    the system to BLOCK high-risk transactions pre-completion.
    """
    db = get_db_connection()
    tx_data = req.model_dump()
    if not tx_data.get("timestamp"):
        tx_data["timestamp"] = datetime.datetime.now().isoformat()

    # ── Init services ────────────────────────────────────────────────────────
    profile_service = ProfileService(db)
    graph_service   = GraphService(db)
    risk_engine     = RiskEngine(db, profile_service, graph_service)

    # ── 1. Structured Risk Pipeline ──────────────────────────────────────────
    pipeline_result  = risk_engine.calculate_risk(req.user_id, tx_data)
    raw_score        = pipeline_result["risk_score"]        # 0-100 int
    raw_reasons      = pipeline_result["reasons"]
    component_scores = pipeline_result["component_scores"]  # per-layer dict

    # ── 2. Decision Engine (returns 0-1 score + structured decision) ─────────
    decision_result = DecisionEngine.decide(raw_score, raw_reasons, component_scores)
    risk_score_norm = decision_result["risk_score"]  # 0.0–1.0
    decision        = decision_result["decision"]    # APPROVE / MFA_HOLD / BLOCK
    reasons         = decision_result["reasons"]
    severity        = decision_result["severity"]

    # ── 3. Profile Updates ───────────────────────────────────────────────────
    profile_service.update_profile(req.user_id, tx_data)
    record_transaction(req.user_id, tx_data)

    # ── 4. Persist to DB ─────────────────────────────────────────────────────
    try:
        db.execute(
            """INSERT INTO transactions
               (tx_id, user_id, amount, merchant_id, timestamp, decision, risk_score, flags, component_scores, city, device_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                req.tx_id, req.user_id, req.amount, req.merchant_id,
                tx_data["timestamp"], decision,
                raw_score,
                json.dumps(reasons),
                json.dumps(component_scores),
                req.city,
                req.device_id,
                datetime.datetime.now()
            ),
        )
        db.commit()
    except Exception as e:
        print("DB Insert Error:", e)

    # ── 5. Offload LLM Case File for high-risk decisions ────────────────────
    if decision in ["BLOCK", "MFA_HOLD"]:
        background_tasks.add_task(
            llm_case_file.generate_case_file, req.tx_id, tx_data, raw_score, decision, reasons
        )

    # ── 6. Chain state snapshot ──────────────────────────────────────────────
    current_chain = graph_service.get_current_chain_state(req.user_id)

    db.close()

    return {
        # ── Identity
        "transaction_id":  req.tx_id,
        "tx_id":           req.tx_id,
        # ── Normalised score (0-1) as required
        "risk_score":      risk_score_norm,
        # ── Raw pipeline score (0-100) for internal use / display
        "raw_score":       raw_score,
        # ── Decision payload
        "decision":        decision,
        "severity":        severity,
        "reasons":         reasons,
        "explanation":     decision_result.get("explanation", ""),
        "flags":           reasons,
        "ml_risk_score":   component_scores.get("ml", 0),
        "fraud_probability": component_scores.get("_fraud_probability", 0.0),
        "ml_explanations": component_scores.get("_ml_explanations", []),
        # ── Score breakdown (NEW: for frontend panels)
        "ml_score":        decision_result.get("ml_score", 0.0),
        "rule_score":      decision_result.get("rule_score", 0.0),
        "final_score":     decision_result.get("final_score", 0.0),
        # ── Per-layer breakdown (deterministic, no random values)
        "component_scores": {
            "device_score":    component_scores.get("device", 0),
            "behavioral_score": component_scores.get("behavioral", 0),
            "ml_score":        component_scores.get("ml", 0),
            "chain_score":     component_scores.get("chain", 0),
            "amount_score":    component_scores.get("amount", 0),
            "location_score":  component_scores.get("location", 0),
        },
        # ── Chain state
        "chain_state":     current_chain,
        # ── Pre-transaction gate flag
        "pre_tx_blocked":  decision == "BLOCK",
    }


# ============================================================================
# ENDPOINT 1b — POST /api/pretransaction  (pre-completion risk gate)
# ============================================================================

@router.post("/api/pretransaction")
async def pre_transaction_check(req: TransactionRequest):
    """
    Lightweight pre-transaction risk gate — evaluates risk without persisting
    or triggering background tasks. Designed for <300ms response time.
    Call this BEFORE committing the transaction to decide whether to proceed.
    """
    db = get_db_connection()
    tx_data = req.model_dump()
    if not tx_data.get("timestamp"):
        tx_data["timestamp"] = datetime.datetime.now().isoformat()

    profile_service = ProfileService(db)
    graph_service   = GraphService(db)
    risk_engine     = RiskEngine(db, profile_service, graph_service)

    pipeline_result = risk_engine.calculate_risk(req.user_id, tx_data)
    raw_score       = pipeline_result["risk_score"]
    raw_reasons     = pipeline_result["reasons"]
    component_scores = pipeline_result["component_scores"]

    decision_result = DecisionEngine.decide(raw_score, raw_reasons, component_scores)
    db.close()

    return {
        "tx_id":            req.tx_id,
        "risk_score":       decision_result["risk_score"],
        "decision":         decision_result["decision"],
        "severity":         decision_result["severity"],
        "reasons":          decision_result["reasons"],
        "pre_tx_blocked":   decision_result["decision"] == "BLOCK",
        "ml_risk_score":    component_scores.get("ml", 0),
        "fraud_probability": component_scores.get("_fraud_probability", 0.0),
        "ml_explanations":  component_scores.get("_ml_explanations", []),
        "component_scores": {
            "device_score":     component_scores.get("device", 0),
            "behavioral_score": component_scores.get("behavioral", 0),
            "ml_score":         component_scores.get("ml", 0),
            "chain_score":      component_scores.get("chain", 0),
            "amount_score":     component_scores.get("amount", 0),
            "location_score":   component_scores.get("location", 0),
        },
    }


# ============================================================================
# ENDPOINT 2 — POST /api/mfa/verify
# ============================================================================

@router.post("/api/mfa/verify")
async def verify_mfa(req: MFARequest):
    db = get_db_connection()

    tx_row = db.execute(
        "SELECT * FROM transactions WHERE tx_id = ?",
        (req.tx_id,),
    ).fetchone()

    if not tx_row:
        raise HTTPException(status_code=404, detail="Transaction not found")

    if req.otp_code != VALID_OTP:
        chain_svc = GraphService(db)
        chain_state = chain_svc.process_chain_event(tx_row["user_id"], "FAILED_MFA")
        db.execute(
            "UPDATE transactions SET chain_state = ? WHERE tx_id = ?",
            (chain_state, req.tx_id),
        )
        db.commit()
        db.close()
        return {"status": "failed", "chain_state": chain_state}

    final_decision = "APPROVE"
    purpose_verdict = None

    if req.purpose and hasattr(llm_case_file, "classify_transfer_purpose"):
        purpose_result = await llm_case_file.classify_transfer_purpose(req.purpose)
        purpose_verdict = purpose_result.get("classification")

        if purpose_verdict == "CONFIRMED_SCAM":
            final_decision = "BLOCK"
            chain_svc = GraphService(db)
            chain_state = chain_svc.process_chain_event(tx_row["user_id"], "FAILED_MFA")
        elif purpose_verdict == "POSSIBLE_SCAM":
            existing_case = tx_row["case_file"] or ""
            warning = (
                f"\n\n⚠️  PURPOSE REVIEW WARNING: User stated '{req.purpose}'. "
                "LLM classification: POSSIBLE_SCAM. Manual review advised."
            )
            db.execute(
                "UPDATE transactions SET case_file = ? WHERE tx_id = ?",
                (existing_case + warning, req.tx_id),
            )
            chain_svc = GraphService(db)
            chain_state = chain_svc.process_chain_event(tx_row["user_id"], "MFA_SUCCESS")
        else:
            chain_svc = GraphService(db)
            chain_state = chain_svc.process_chain_event(tx_row["user_id"], "MFA_SUCCESS")
    else:
        chain_svc = GraphService(db)
        chain_state = chain_svc.process_chain_event(tx_row["user_id"], "MFA_SUCCESS")

    db.execute(
        "UPDATE transactions SET decision = ?, chain_state = ? WHERE tx_id = ?",
        (final_decision, chain_state, req.tx_id),
    )
    db.commit()
    db.close()

    return {
        "status":           "verified",
        "final_decision":   final_decision,
        "chain_state":      chain_state,
        "purpose_verdict":  purpose_verdict,
    }


# ============================================================================
# ENDPOINT 3 — POST /api/analyst/confirm
# ============================================================================

@router.post("/api/analyst/confirm")
async def analyst_confirm(req: AnalystConfirmRequest):
    db = get_db_connection()

    tx_row = db.execute(
        "SELECT * FROM transactions WHERE tx_id = ?",
        (req.tx_id,),
    ).fetchone()

    if not tx_row:
        raise HTTPException(status_code=404, detail="Transaction not found")

    feature_vector = json.dumps({
        "amount":                  tx_row["amount"],
        "balance_diff_orig":       tx_row["oldbalanceOrg"] - tx_row["newbalanceOrig"]
                                   if "oldbalanceOrg" in tx_row.keys() else 0,
        "balance_diff_dest":       tx_row["newbalanceDest"] - tx_row["oldbalanceDest"]
                                   if "newbalanceDest" in tx_row.keys() else 0,
        "hour_of_day":             datetime.datetime.fromisoformat(tx_row["timestamp"]).hour,
        "tx_type_encoded":         _encode_tx_type(tx_row.get("tx_type", "TRANSFER")),
        "amount_to_balance_ratio": (
            tx_row["amount"] / tx_row["oldbalanceOrg"]
            if tx_row.get("oldbalanceOrg") and tx_row["oldbalanceOrg"] > 0
            else 0
        ),
    })

    db.execute(
        """INSERT OR REPLACE INTO retraining_buffer
            (tx_id, feature_vector, is_fraud, analyst_id, timestamp)
        VALUES (?, ?, ?, ?, ?)""",
        (
            req.tx_id,
            feature_vector,
            int(req.is_fraud),
            req.analyst_id,
            datetime.datetime.now().isoformat(),
        ),
    )

    if req.is_fraud:
        chain_svc = GraphService(db)
        chain_svc.process_chain_event(tx_row["user_id"], "ANALYST_CONFIRM_FRAUD")

    db.commit()
    db.close()
    return {"status": "recorded", "tx_id": req.tx_id}


# ============================================================================
# ENDPOINT 5 — GET /api/case/{tx_id}
# ============================================================================

@router.get("/api/case/{tx_id}")
async def get_case_by_tx_id(tx_id: str):
    db  = get_db_connection()
    row = db.execute(
        "SELECT case_file FROM transactions WHERE tx_id = ?",
        (tx_id,),
    ).fetchone()
    db.close()

    if not row:
        raise HTTPException(status_code=404, detail="Transaction not found")

    case_file = row["case_file"]
    if case_file in (None, "PENDING"):
        return {"status": "pending"}

    return {"status": "ready", "case_file": case_file}


# ============================================================================
# ENDPOINT 6 — GET /api/chain/{user_id}
# ============================================================================

@router.get("/api/chain/{user_id}")
async def get_chain(user_id: str):
    db  = get_db_connection()
    row = db.execute(
        "SELECT state, event_log, suspicion_score FROM chain_states WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    db.close()

    if not row:
        return {"state": "CLEAR", "event_log": [], "suspicion_score": 0}

    try:
        event_log = json.loads(row["event_log"] or "[]")
    except (json.JSONDecodeError, TypeError):
        event_log = []

    return {
        "state":           row["state"],
        "event_log":       event_log,
        "suspicion_score": row["suspicion_score"],
    }


# ============================================================================
# ENDPOINT 7 — POST /api/chain/reset
# ============================================================================

@router.post("/api/chain/reset")
async def reset_chain(req: ChainResetRequest):
    db = get_db_connection()
    chain_svc = GraphService(db)
    chain_svc.reset_chain(req.user_id)
    db.close()
    return {"status": "reset", "user_id": req.user_id}


# ============================================================================
# ENDPOINT 8 — GET /api/metrics
# ============================================================================

@router.get("/api/metrics")
async def get_metrics():
    db    = get_db_connection()
    today = datetime.date.today().isoformat()

    total_tx    = db.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()["cnt"]
    block_count = db.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE timestamp LIKE ? AND decision = 'BLOCK'", (f"{today}%",)).fetchone()["cnt"]
    held_count  = db.execute("SELECT COUNT(*) AS cnt FROM transactions WHERE timestamp LIKE ? AND decision = 'MFA_HOLD'", (f"{today}%",)).fetchone()["cnt"]
    fraud_rate  = round(((block_count + held_count) / total_tx * 100) if total_tx > 0 else 0.0, 2)
    avg_risk    = db.execute("SELECT AVG(risk_score) AS avg FROM transactions WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()["avg"] or 0.0

    model_version = db.execute("SELECT COALESCE(MAX(version), 1) AS ver FROM model_metrics").fetchone()["ver"]
    db.close()

    simulated_latency_ms = round(8 + total_tx * 0.25, 1)
    throughput_tps       = round(total_tx / max(1, 600), 4)

    return {
        "total_tx":             total_tx,
        "fraud_rate":           fraud_rate,
        "avg_risk":             round(avg_risk, 2),
        "model_version":        model_version,
        "fraud_detection_rate": fraud_rate,
        "avg_latency_ms":       simulated_latency_ms,
        "throughput_tps":       throughput_tps,
        "total_transactions":   total_tx,
    }


# ============================================================================
# ENDPOINT 9 — POST /api/coordinated/check
# ============================================================================

@router.post("/api/coordinated/check")
async def check_coordinated(req: CoordinatedCheckRequest):
    db     = get_db_connection()
    cutoff = (datetime.datetime.now() - datetime.timedelta(seconds=90)).isoformat()

    count = db.execute(
        "SELECT COUNT(*) AS cnt FROM transactions WHERE merchant_id = ? AND timestamp >= ?",
        (req.merchant_id, cutoff),
    ).fetchone()["cnt"]
    db.close()

    if count >= 5:
        return {"coordinated": True, "attack_size": count, "merchant_id": req.merchant_id}
    return {"coordinated": False, "merchant_id": req.merchant_id}


# ============================================================================
# ENDPOINT 10 — GET /api/drift/alerts
# ============================================================================

@router.get("/api/drift/alerts")
async def get_drift_alerts():
    db   = get_db_connection()
    rows = db.execute("SELECT * FROM drift_alerts ORDER BY detected_at DESC LIMIT 10").fetchall()
    db.close()
    return [dict(r) for r in rows]


# ============================================================================
# ENDPOINT 11 — POST /api/scenario/{scenario_name}
# ============================================================================

@router.post("/api/scenario/{scenario_name}")
async def run_scenario(scenario_name: str):
    txs = SCENARIOS.get(scenario_name.lower())
    if not txs:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIOS.keys())}",
        )

    completed_tx_ids: list[str] = []

    for tx in txs:
        tx["timestamp"] = datetime.datetime.now().isoformat()
        tx_req = TransactionRequest(**tx)
        bt     = BackgroundTasks()
        await process_transaction(tx_req, bt)

        for task in bt.tasks:
            try:
                if asyncio.iscoroutinefunction(task.func):
                    await task.func(*task.args, **task.kwargs)
                else:
                    task.func(*task.args, **task.kwargs)
            except Exception as exc:
                print(f"Scenario background task failed: {exc}")

        completed_tx_ids.append(tx["tx_id"])
        await asyncio.sleep(1.5)

    return {"status": "completed", "scenario": scenario_name, "transactions": completed_tx_ids}


# ============================================================================
# ENDPOINT 12 — GET /api/cases
# ============================================================================

@router.get("/api/cases")
async def list_cases():
    cases = []
    for key, filename in CASE_FILES.items():
        filepath = CASE_CACHE_DIR / filename
        cases.append({"key": key, "filename": filename, "available": filepath.exists()})
    return {"cases": cases, "total": len(cases)}


# ============================================================================
# ENDPOINT 13 — GET /api/cases/{case_key}
# ============================================================================

@router.get("/api/cases/{case_key}")
async def get_case_by_key(case_key: str):
    if case_key not in CASE_FILES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown case key '{case_key}'. Valid keys: {list(CASE_FILES.keys())}",
        )

    filepath = CASE_CACHE_DIR / CASE_FILES[case_key]
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Case file not found on disk.")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Malformed JSON in case file: {e}")

    return {"key": case_key, "filename": CASE_FILES[case_key], "case_file": data["case_file"]}


# ============================================================================
# ENDPOINT 14 — GET /api/cases/bulk/all
# ============================================================================

@router.get("/api/cases/bulk/all")
async def get_all_cases():
    results = {}
    for key, filename in CASE_FILES.items():
        filepath = CASE_CACHE_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[key] = {"filename": filename, "case_file": data["case_file"]}
        else:
            results[key] = {"filename": filename, "case_file": None, "error": "File not found"}
    return {"cases": results}


# ============================================================================
# ENDPOINT 15 — GET /api/coordinated/{merchant_id}
# ============================================================================

@router.get("/api/coordinated/{merchant_id}")
async def check_coordinated_get(merchant_id: str):
    """GET alias so the frontend can poll without sending a request body."""
    db     = get_db_connection()
    cutoff = (datetime.datetime.now() - datetime.timedelta(seconds=90)).isoformat()

    count = db.execute(
        "SELECT COUNT(*) AS cnt FROM transactions WHERE merchant_id = ? AND timestamp >= ?",
        (merchant_id, cutoff),
    ).fetchone()["cnt"]
    unique_users = db.execute(
        "SELECT COUNT(DISTINCT user_id) AS cnt FROM transactions WHERE merchant_id = ? AND timestamp >= ?",
        (merchant_id, cutoff),
    ).fetchone()["cnt"]
    db.close()

    if count >= 3:
        return {
            "coordinated":    True,
            "attack_size":    unique_users,
            "merchant_id":    merchant_id,
            "shared_devices": max(0, unique_users - count),
        }
    return {"coordinated": False, "merchant_id": merchant_id}


# ============================================================================
# ENDPOINT 16 — POST /api/scenario/run  (body-based alias)
# ============================================================================

@router.post("/api/scenario/run")
async def run_scenario_body(req: ScenarioRunRequest):
    """Body-based scenario runner — delegates to the path handler."""
    return await run_scenario(req.scenario)


# ============================================================================
# ENDPOINT 17 — REMOVED (Moved to main.py for adaptive feedback logic)
# ============================================================================


# ============================================================================
# ENDPOINT 18 — GET /api/metrics/live
# ============================================================================

@router.get("/api/metrics/live")
async def get_live_metrics():
    """Extended live metrics for the command center dashboard."""
    db    = get_db_connection()
    today = datetime.date.today().isoformat()

    total    = db.execute("SELECT COUNT(*) AS c FROM transactions WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()["c"]
    blocked  = db.execute("SELECT COUNT(*) AS c FROM transactions WHERE timestamp LIKE ? AND decision='BLOCK'", (f"{today}%",)).fetchone()["c"]
    held     = db.execute("SELECT COUNT(*) AS c FROM transactions WHERE timestamp LIKE ? AND decision='MFA_HOLD'", (f"{today}%",)).fetchone()["c"]
    avg_risk = db.execute("SELECT AVG(risk_score) AS a FROM transactions WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()["a"] or 0
    db.close()

    fraud_rate = round((blocked + held) / total * 100, 2) if total > 0 else 0.0
    return {
        "total_transactions":   total,
        "fraud_detection_rate": fraud_rate,
        "avg_latency_ms":       round(12 + total * 0.3, 1),
        "throughput_tps":       round(total / max(1, 300), 4),
        "blocked":              blocked,
        "held":                 held,
        "avg_risk_score":       round(avg_risk, 2),
    }


# ============================================================================
# ENDPOINT 19 — GET /api/simulate
# ============================================================================

@router.get("/api/simulate")
async def simulate_single(profile: str = "random", background_tasks: BackgroundTasks = None):
    """
    Generate and process a single simulated transaction.
    profile: 'safe' | 'mfa' | 'block' | 'random'
    Distribution (random): 40% SAFE · 35% MFA · 25% BLOCK
    """
    tx_data = generate_single(profile)
    # Strip internal _profile key before passing to pipeline
    tx_data.pop("_profile", None)

    bg = BackgroundTasks()
    try:
        tx_req = TransactionRequest(**tx_data)
        result = await process_transaction(tx_req, bg)
        # Combine the original generated payload (tx_data) with the evaluation result
        return {"simulated": True, "profile": profile, **tx_data, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


# ============================================================================
# ENDPOINT 20 — POST /api/simulate/batch
# ============================================================================

class SimulateBatchRequest(BaseModel):
    count: int = 10
    profile: str = "random"   # 'safe' | 'mfa' | 'block' | 'random'


@router.post("/api/simulate/batch")
async def simulate_batch(req: SimulateBatchRequest):
    """
    Generate and process a batch of simulated transactions.
    Returns summary counts by decision alongside full results.
    Max 50 per call to keep response snappy.
    """
    count   = min(req.count, 50)
    results = []
    summary = {"APPROVE": 0, "MFA_HOLD": 0, "BLOCK": 0}

    for tx_data in generate_batch(count):
        expected_profile = tx_data.pop("_profile", "random")
        # Override profile if user specified one
        if req.profile != "random":
            from backend.services.simulator import generate_single
            tx_data = generate_single(req.profile)
            tx_data.pop("_profile", None)

        try:
            tx_req = TransactionRequest(**tx_data)
            result = await process_transaction(tx_req, BackgroundTasks())
            decision = result.get("decision", "APPROVE")
            summary[decision] = summary.get(decision, 0) + 1
            results.append({
                "tx_id":           result["tx_id"],
                "expected_profile": expected_profile,
                "decision":        decision,
                "risk_score":      result["risk_score"],
                "ml_risk_score":   result.get("ml_risk_score", 0),
                "fraud_probability": result.get("fraud_probability", 0.0),
                "reasons":         result["reasons"][:3],
            })
        except Exception as e:
            results.append({"tx_id": tx_data.get("tx_id"), "error": str(e)})

    return {
        "simulated": True,
        "count":     count,
        "summary":   summary,
        "results":   results,
    }