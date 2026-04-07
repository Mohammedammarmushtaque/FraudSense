<h1>🛡️ FraudSense: Real-Time Fraud Detection System</h1>

<p>
<b>FraudSense</b> is a high-performance, multi-layered fraud detection system designed to identify and block sophisticated financial fraud patterns in real-time.
</p>

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/6fd47ebd-536b-4465-bf79-4724c6e51835" />

<p>
Built using a hybrid approach of rule-based intelligence, behavioral profiling, and machine learning, the system focuses on detecting high-risk activities such as Account Takeovers (ATO), coordinated fraud rings, and abnormal transaction behavior.
</p>

<p>
🏆 <b>HackUp 2026:</b> Selected in the <b>Top 45 teams</b> out of 200 teams.
</p>

<hr>

<h2>🚀 Core Features</h2>

<h3>🛡️ 6-Layer Risk Engine</h3>
<ul>
<li>Location anomalies</li>
<li>Device trust</li>
<li>Behavioral patterns</li>
<li>ML-based probability</li>
<li>Network graph connections</li>
<li>Transaction chain patterns</li>
</ul>

<h3>📈 Dominant Signal Scoring</h3>
<p>Custom scoring logic ensures high-risk signals are never diluted:</p>
<pre>
FinalScore = (0.7 × MaxLayer) + (0.3 × WeightedAvg)
</pre>

<h3>🔄 Adaptive Risk Logic</h3>
<p>
Dynamically adjusts risk sensitivity based on observed transaction patterns and system feedback.
</p>

<h3>⚡ Graph-Based Fraud Detection</h3>
<p>
Detects circular money flows and coordinated attacks using NetworkX graph analysis.
</p>

<h3>🧠 Hybrid Intelligence System</h3>
<p>
Combines deterministic rules with ML models (Random Forest / XGBoost) for robust detection.
</p>

<hr>

<h2>🏗️ Architecture</h2>

<ul>
<li><b>Backend:</b> FastAPI (Python)</li>
<li><b>Frontend:</b> React + Vite (Command Center Dashboard)</li>
<li><b>ML Engine:</b> Scikit-learn (Random Forest / XGBoost)</li>
<li><b>Graph Engine:</b> NetworkX</li>
<li><b>Database:</b> SQLite</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
backend/          # Risk Engine, Services, Graph Logic
frontend/         # React Dashboard
models/           # Trained ML models (.pkl)
main.py           # Entry point
requirements.txt  # Dependencies
README.md         # Documentation
</pre>

<hr>

<h2>🛠️ Setup & Installation</h2>

<h3>1. Backend</h3>
<pre>
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
</pre>

<h3>2. Frontend</h3>
<pre>
cd frontend
npm install
npm run dev
</pre>

<hr>

<h2>📡 API Endpoint</h2>

<h3>💳 Process Transaction</h3>
<p><b>POST /api/transaction</b></p>

<p>Evaluates a transaction and returns a fraud risk decision.</p>

<h4>Sample Input</h4>
<pre>
{
  "tx_id": "TX-9901",
  "user_id": "user_1001",
  "amount": 85000.0,
  "city": "Mumbai",
  "device_id": "DEV-SAFE-01",
  "tx_type": "TRANSFER",
  "channel": "mobile"
}
</pre>

<h4>Sample Response</h4>
<pre>
{
  "status": "success",
  "decision": "APPROVE",
  "risk_score": 12
}
</pre>

<hr>

<h2>🧠 Model Training</h2>

<p>To retrain the fraud detection model:</p>

<pre>
python train_models.py
</pre>

<hr>

<h2>📊 Dashboard</h2>

<p>The <b>FraudSense Command Center</b> provides:</p>
<ul>
<li>Real-time transaction monitoring</li>
<li>Risk score visualization</li>
<li>Fraud pattern tracking</li>
<li>System performance insights</li>
</ul>

<p>Runs locally via Vite after frontend setup.</p>

<hr>

<h2>🎯 Key Highlights</h2>

<ul>
<li>Designed for real-time fraud detection use cases</li>
<li>Strong focus on system design and scoring logic</li>
<li>Built during a competitive hackathon environment</li>
<li>Clean modular architecture for scalability</li>
</ul>

<hr>

<hr>

<h2>🤝 Contributing</h2>

<p>
I would like to welcome contributions to improve FraudSense 🚀
</p>

<ul>
<li>Fix bugs or improve performance</li>
<li>Add new fraud detection features</li>
<li>Enhance UI/UX of the dashboard</li>
<li>Improve documentation</li>
</ul>

<h3>📌 How to Contribute</h3>

<ol>
<li>Fork the repository</li>
<li>Create a new branch (<code>feature/your-feature-name</code>)</li>
<li>Make your changes</li>
<li>Commit and push</li>
<li>Open a Pull Request</li>
</ol>

<p>
For major changes, please open an issue first to discuss what you would like to change.
</p>

<p>© 2026 FraudSense Team</p>
