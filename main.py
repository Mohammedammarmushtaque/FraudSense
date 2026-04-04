# main.py
# FraudSense FastAPI application — all endpoints wired together
# Run with: uvicorn main:app --reload --port 8000

import asyncio
import json
import sqlite3
import datetime
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Project modules ──────────────────────────────────────────────────────────
from backend.db.database import initialize_database, get_db_connection
from backend.services.graph_service import GraphService
from backend.api.transaction import router as transaction_router, process_transaction, TransactionRequest
import llm_case_file
from retrain import RetrainWorker


# ============================================================================
# Constants and Configuration
# ============================================================================

VALID_OTP = "1234"
CASE_CACHE_DIR = Path("case_cache")

# Map of friendly case keys → filenames
CASE_FILES = {
    "ato":             "ato_case.json",
    "coordinated":     "coordinated_case.json",
    "mimicry":         "mimicry_case.json",
    "scam_romance":    "scam_romance.json",
    "stealth_probe":   "stealth_probe_case.json",
}


# ============================================================================
# Startup / shutdown lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables and launch the background retraining worker."""
    initialize_database()             # create all tables if absent

    # Ensure necessary tables exist
    _db = get_db_connection()
    _db.execute("""
        CREATE TABLE IF NOT EXISTS fraud_list (
            user_id TEXT,
            device_id TEXT,
            reason TEXT,
            added_at TIMESTAMP
        )
    """)
    _db.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            tx_id TEXT,
            is_fraud INTEGER,
            created_at TIMESTAMP
        )
    """)
    _db.commit()
    _db.close()

    # Start the adaptive retraining daemon
    worker = RetrainWorker("fraudsense.db")
    worker.start()
    app.state.retrain_worker = worker

    print("FraudSense API running on http://localhost:8000")
    yield

    # Graceful shutdown — signal the worker to stop
    worker.stop()
    worker.join(timeout=10)


# ============================================================================
# Application bootstrap
# ============================================================================

app = FastAPI(
    title="FraudSense",
    description="Real-time fraud detection with chain-state and LLM case files",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

# ============================================================================
# Shared singletons
# ============================================================================

def get_db() -> sqlite3.Connection:
    """Return a Row-factory enabled DB connection."""
    return get_db_connection()

app.include_router(transaction_router)


# ============================================================================
# Adaptive Learning Weights API  🔥 UPDATED
# ============================================================================

@app.get("/api/weights")
def get_weights():
    """
    Returns weights adjusted based on analyst feedback.
    Analyzes historical feedback to recalibrate layer importance.
    """
    conn = get_db_connection()
    # Query last 50 feedback records
    feedback = conn.execute(
        "SELECT is_fraud FROM feedback ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()

    total = len(feedback)
    fraud_count = sum(1 for f in feedback if f["is_fraud"] == 1)
    fraud_ratio = fraud_count / total if total > 0 else 0

    base_weights = {
        "device": 15,
        "behavioral": 25,
        "ml": 25,
        "graph": 15,
        "chain": 20
    }

    # 🔥 Adaptive Logic: If high fraud detected in feedback, shift weight to behavioral/ml
    if fraud_ratio > 0.6:
        base_weights["ml"] += 5
        base_weights["behavioral"] += 5
        base_weights["graph"] += 3
    elif fraud_ratio > 0.3:
        base_weights["ml"] += 2
        base_weights["behavioral"] += 2

    # Ensure weights stay between 5 and 50 and add small demo randomization
    import random
    final_weights = {
        k: max(5, min(50, v + random.randint(-2, 2)))
        for k, v in base_weights.items()
    }

    return final_weights

@app.post("/api/feedback")
def record_feedback(data: dict):
    """
    Records analyst feedback on whether a transaction was correctly identified.
    Triggers adaptive weight updates.
    """
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO feedback (tx_id, is_fraud, created_at) VALUES (?, ?, ?)",
        (data["tx_id"], data["is_fraud"], datetime.datetime.now())
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/transactions")
def list_transactions():
    """Returns the last 50 transactions for the live feed."""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT * FROM transactions ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ============================================================================
# Case Generation Endpoints
# ============================================================================

@app.post("/api/case")
def generate_case(data: dict):
    """
    Mock AI Case Generator endpoint.
    """
    return {
        "summary": "High-risk transaction detected",
        "risk_factors": [
            "Unusual location",
            "New device",
            "High transaction amount"
        ],
        "decision": "BLOCK"
    }