# 🛡️ FraudSense: Real-Time Fraud Detection System

**FraudSense** is a high-performance, multi-layered fraud detection pipeline designed to identify and block sophisticated financial fraud patterns in real-time. By combining rule-based engines, predictive machine learning, and Generative AI for explainability, FraudSense provides a production-grade defense against ATO (Account Takeover), Coordinated Ring Attacks, and Social Engineering scams.

🏆 **HackUp 2026:** Selected in the **Top 45 teams** out of 5,000+ applicants.

---

## 🚀 Core Features

- **🛡️ 6-Layer Risk Engine:** Evaluates transactions across Location, Device, Behavioral Profile, ML Probability, Network Graphs, and Chain-State.
- **📈 Dominant Signal Scoring:** Uses a custom non-averaging logic $FinalScore = (0.7 \times MaxLayer) + (0.3 \times WeightedAvg)$ to ensure high-risk signals are never "averaged down."
- **🤖 GenAI Explainability:** Integrated with Google Gemini (and local Ollama fallback) to generate human-readable investigation reports for every flagged transaction.
- **🔄 Adaptive Learning Loop:** Automatically recalibrates layer weights based on historical analyst feedback to stay ahead of evolving fraud tactics.
- **⚡ Real-Time Graph Intelligence:** Detects circular money laundering flows and coordinated merchant targeting via NetworkX graph analysis.

---

## 🏗️ Architecture

FraudSense is built on a modern, asynchronous architecture designed for sub-300ms latency:

- **Backend:** FastAPI (Python) for asynchronous request handling.
- **Frontend:** React + Vite with Framer Motion for a high-fidelity "Command Center" dashboard.
- **Engine:** Hybrid approach (Deterministic Rules + Random Forest/XGBoost).
- **Persistence:** SQLite with persistent caching for millisecond user profile lookups.
- **Background Tasks:** Adaptive retraining and AI report generation run asynchronously to prevent blocking the transaction flow.

---

## 📂 Project Structure

```text
backend/          # API, Services (Risk Engine, Behavioral, Graph)
frontend/         # React Dashboard (Vite)
models/           # Pre-trained ML models (.pkl)
main.py           # Application Entry Point
requirements.txt  # Project Dependencies
README.md         # Professional Documentation
```

---

## 🛠️ Setup & Installation

### 1. Backend (Python 3.9+)
Install dependencies and run the server:
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API (on http://localhost:8000)
uvicorn main:app --reload --port 8000
```

### 2. Frontend (React/Vite)
Build and run the dashboard:
```bash
cd frontend
npm install
npm run dev
```

### 🔧 Configuration
Create a `.env` file in the root if you wish to use GenAI features:
```text
GEMINI_API_KEY=your_key_here
```
*Note: If no API key is provided, the system falls back to a mock AI generator or local Ollama instance.*

---

## 📡 API Endpoints

### 💳 Process Transaction
`POST /api/transaction`
Primary endpoint for real-time risk assessment.

**Sample Request Body:**
```json
{
  "tx_id": "TX-9901",
  "user_id": "user_1001",
  "amount": 85000.0,
  "city": "Mumbai",
  "device_id": "DEV-SAFE-01",
  "tx_type": "TRANSFER",
  "channel": "mobile"
}
```

**Response:**
```json
{
  "status": "success",
  "decision": "APPROVE",
  "risk_score": 12,
  "case_file": "..."
}
```

---

## 🧠 ML Model Generation
If `models/fraud_model.pkl` is missing, you can regenerate the synthetic data and retrain the model by running:
```bash
python train_models.py
```

---

## 📊 Dashboard Interface (localhost:3000)
The **FraudSense Command Center** provides a mission-critical view of system health, real-time transaction bursts, and adaptive weight tuning.

---
© 2026 FraudSense Team | Developed for HackUp 2026.
