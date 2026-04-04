// App.jsx — FraudSense Command Center
// Full overhaul: Fraud Command Center layout with all panels

import React, { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// ── New panel components ─────────────────────────────────────────────────────
import RiskPanel          from "./components/RiskPanel";
import TriggerInsights    from "./components/TriggerInsights";
import AIDecisionPanel    from "./components/AIDecisionPanel";
import ChainView          from "./components/ChainView";
// ── Audio utility ────────────────────────────────────────────────────────────
import { playTick, playAlert } from "./utils/audio";
import GraphAlert         from "./components/GraphAlert";
import PipelineBreakdown  from "./components/PipelineBreakdown";
import AdaptivePanel      from "./components/AdaptivePanel";
import CaseFilePanel       from "./components/CaseFilePanel";
import MetricsPanel       from "./components/MetricsPanel";
import TransactionWindow  from "./components/TransactionWindow";
import TransactionSimulator from "./components/TransactionSimulator";

// ── Legacy components (kept for their logic) ─────────────────────────────────
import MFAModal           from "./components/MFAModal";
import DemoControls       from "./components/DemoControls";

const API = "/api";  // Vite proxy → localhost:8000

function useInterval(callback, delay) {
  const saved = useRef(callback);
  useEffect(() => { saved.current = callback; }, [callback]);
  useEffect(() => {
    if (delay == null) return;
    const id = setInterval(() => saved.current(), delay);
    return () => clearInterval(id);
  }, [delay]);
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

// ── Scenario runner (calls /api/scenario/{name} directly) ─────────────────
async function runScenario(name) {
  const SCENARIO_MAP = {
    ato_attack:          "ato",
    coordinated_attack:  "coordinated",
    mimicry_attack:      "mimicry",
    app_scam:            "scam",
    stealth_probe:       "stealth_probe",
    ato:                 "ato",
    coordinated:         "coordinated",
    mimicry:             "mimicry",
    scam:                "scam",
  };
  const backendName = SCENARIO_MAP[name] || name;
  return apiFetch(`/scenario/${backendName}`, { method: "POST" });
}

async function submitTransaction(payload) {
  return apiFetch("/transaction", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

// ── Decision colouring helpers ────────────────────────────────────────────────
function decisionColor(decision) {
  if (decision === "BLOCK")    return "text-red-400 bg-red-900/30 border-red-700/50";
  if (decision === "MFA_HOLD") return "text-amber-400 bg-amber-900/30 border-amber-700/50";
  return "text-emerald-400 bg-emerald-900/20 border-emerald-700/40";
}

function decisionDot(decision) {
  if (decision === "BLOCK")    return "bg-red-500";
  if (decision === "MFA_HOLD") return "bg-amber-500 animate-pulse";
  return "bg-emerald-500";
}

// ── Small transaction list item ───────────────────────────────────────────────

// ── Feedback Handler ────────────────────────────────────────────────────────
const recordFeedback = async (txId, isFraud) => {
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tx_id: txId, is_fraud: isFraud }),
    });
    // Trigger weight refresh
    window.dispatchEvent(new Event("weights-update"));
  } catch (err) {
    console.error("Feedback error:", err);
  }
};

function TxRow({ tx, isSelected, onClick }) {
  const dc = decisionColor(tx.decision);
  const dot = decisionDot(tx.decision);
  const risk = tx.risk_score ?? 0;
  const riskColor = risk >= 66 ? "text-red-400" : risk >= 36 ? "text-amber-400" : "text-emerald-400";

  return (
    <div className={`group relative w-full flex flex-col border-b border-gray-800/60 transition-all duration-150
      hover:bg-gray-800/80 ${isSelected ? "bg-gray-800/90 border-l-2 border-l-indigo-500" : "border-l-2 border-l-transparent"}`}>
      
      <button
        onClick={() => onClick(tx)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left focus:outline-none"
      >
        <span className={`h-2.5 w-2.5 rounded-full flex-shrink-0 ${dot}`} />
        <div className="flex-1 min-w-0">
          <p className="text-xs font-mono font-semibold text-gray-300 truncate flex items-center gap-2">
            {tx.tx_id}
            {tx._simType === "NORMAL" && <span className="px-1.5 py-0.5 text-[8px] bg-emerald-900/40 text-emerald-400 border border-emerald-700/50 rounded">NORMAL</span>}
            {tx._simType === "SUSPICIOUS" && <span className="px-1.5 py-0.5 text-[8px] bg-amber-900/40 text-amber-400 border border-amber-700/50 rounded">SUSPICIOUS</span>}
            {tx._simType === "FRAUD" && <span className="px-1.5 py-0.5 text-[8px] bg-red-900/40 text-red-400 border border-red-700/50 rounded">FRAUD</span>}
          </p>
          <p className="text-[10px] text-gray-500 truncate">{tx.user_id} · {tx.merchant_id}</p>
        </div>
        <div className="flex flex-col items-end gap-1 flex-shrink-0">
          <span className={`text-xs font-black tabular-nums ${riskColor}`}>{risk}</span>
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${dc}`}>
            {tx.decision === "MFA_HOLD" ? "HOLD" : tx.decision}
          </span>
        </div>
      </button>

      {/* Feedback Buttons (Visible on hover or if selected) */}
      <div className={`flex gap-1 px-4 pb-2 transition-opacity duration-200 ${isSelected ? "opacity-100" : "opacity-0 group-hover:opacity-100"}`}>
        <button 
          onClick={(e) => { e.stopPropagation(); recordFeedback(tx.tx_id, 1); }}
          className="px-2 py-0.5 bg-red-900/40 text-[9px] font-bold text-red-400 border border-red-700/30 rounded hover:bg-red-900/60 transition-colors"
        >
          MARK FRAUD
        </button>
        <button 
          onClick={(e) => { e.stopPropagation(); recordFeedback(tx.tx_id, 0); }}
          className="px-2 py-0.5 bg-emerald-900/40 text-[9px] font-bold text-emerald-400 border border-emerald-700/30 rounded hover:bg-emerald-900/60 transition-colors"
        >
          MARK SAFE
        </button>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [transactions,     setTransactions]     = useState([]);
  const [selectedTx,       setSelectedTx]       = useState(null);
  const [lastTx,           setLastTx]           = useState(null);
  const [metrics,          setMetrics]          = useState(null);
  const [coordinatedAlert, setCoordinatedAlert] = useState(null);
  const [dynamicWeights,   setDynamicWeights]   = useState(null);
  const [weightHistory,    setWeightHistory]    = useState([]);
  const [mfaOpen,          setMfaOpen]          = useState(false);
  const [mfaTx,            setMfaTx]            = useState(null);
  const [scenarioRunning,  setScenarioRunning]  = useState(false);
  const [attackMode,       setAttackMode]       = useState(null);
  const [systemStatus,     setSystemStatus]     = useState("ACTIVE");
  const [selectedCase,     setSelectedCase]     = useState(null);
  const [activeTab,        setActiveTab]        = useState("command"); // command | feed

  const handleSelect = async (tx) => {
    setSelectedTx(tx);
    setLastTx(tx);
    // Optimistically select, then fetch explanation
    try {
      const res = await fetch("/api/case", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(tx)
      });
      const data = await res.json();
      setSelectedCase(data);
    } catch (err) {
      console.error("Case fetch error:", err);
    }
  };

  // ── Polling helpers ────────────────────────────────────────────────────────
  const fetchTransactions = useCallback(async () => {
    try {
      const data = await apiFetch("/transactions?limit=50");
      setTransactions(prev => {
        const prevMap = new Map(prev.map(t => [t.tx_id, t]));
        // Merge missing transactions from DB without overwriting rich local transactions
        for (const dbTx of data) {
          if (!prevMap.has(dbTx.tx_id)) {
            prevMap.set(dbTx.tx_id, dbTx);
          }
        }
        return Array.from(prevMap.values())
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
          .slice(0, 50);
      });
      // Auto-select latest high-risk tx
      setTransactions(prev => {
        const highRisk = prev.find((t) => t.decision !== "APPROVE");
        if (highRisk && !selectedTx) {
          setSelectedTx(highRisk);
          setLastTx(highRisk);
        }
        // MFA auto-open
        const mfaPending = prev.find((t) => t.decision === "MFA_HOLD" && t.requires_mfa);
        if (mfaPending && !mfaOpen) { setMfaTx(mfaPending); setMfaOpen(true); }
        return prev;
      });
    } catch (e) { console.warn("tx poll:", e.message); }
  }, [selectedTx, mfaOpen]);

  const fetchMetrics = useCallback(async () => {
    try {
      const data = await apiFetch("/metrics");
      setMetrics(data);
    } catch (e) { /* silent */ }
  }, []);

  const fetchWeights = useCallback(async () => {
    try {
      const data = await apiFetch("/weights");
      setDynamicWeights(data);
    } catch (e) { /* silent */ }
  }, []);

  const checkCoordinated = useCallback(async () => {
    const merchants = ["M001", "M002", "M003", "MERCH-TARGET-99"];
    try {
      for (const mid of merchants) {
        const r = await apiFetch(`/coordinated/${mid}`);
        if (r.coordinated) { setCoordinatedAlert(r); return; }
      }
      setCoordinatedAlert(null);
    } catch (e) { /* silent */ }
  }, []);

  useInterval(fetchTransactions, 2000);
  useInterval(fetchMetrics,      4000);
  useInterval(checkCoordinated,  3000);
  useInterval(fetchWeights,      5000);

  useEffect(() => {
    fetchTransactions();
    fetchMetrics();
    checkCoordinated();
    fetchWeights();
  }, []); // eslint-disable-line

  // ── Scenario handling (BURST LOGIC) ─────────────────────────────────────────
  const buildScenarioTx = (type, i) => {
    const base = {
      user_id: `user_${type}_${Math.floor(Math.random() * 1000)}`,
      timestamp: new Date().toISOString(),
      channel: "web", 
      merchant_id: "MERCH-MOCK",
      merchant_category: "retail",
      device_timezone: "UTC",
      oldbalanceOrg: 10000,
      newbalanceOrig: 10000,
      oldbalanceDest: 0,
      newbalanceDest: 0,
      tx_type: "PAYMENT"
    };

    switch(type) {
      case "ato":
        return {
          ...base,
          tx_id: `ATO-${Date.now()}-${i}`,
          amount: 85000 + Math.random() * 20000, // Device mismatch + high risk amount
          device_id: `UNRECOGNIZED-X${i}`, 
          city: "Lagos",                         
          merchant_category: "cryptocurrency",
          tx_type: "TRANSFER"
        };
      case "mimicry":
        return {
          ...base,
          tx_id: `MIMIC-${Date.now()}-${i}`,
          amount: 15 + Math.random() * 30,       // Low amount but suspicious pattern
          device_id: "KNOWN-DEVICE-01",             
          city: "Chicago",
          merchant_category: "groceries",        
          tx_type: "PAYMENT"
        };
      case "scam":
        return {
          ...base,
          tx_id: `SCAM-${Date.now()}-${i}`,
          amount: 120000 + Math.random() * 50000, // High amount + location anomaly
          device_id: "SCAM-BURNER-PHONE", 
          city: "Miami",                          
          merchant_category: "wire_transfer",
          tx_type: "TRANSFER"
        };
      case "coordinated":
        return {
          ...base,
          tx_id: `COORD-${Date.now()}-${i}`,
          user_id: `user_botnet_${i}`,           // Ring attack
          amount: 499,
          merchant_id: "TARGET-CORP-99",         
          device_id: `BOT-NODE-${i}`,
          city: "Mumbai"
        };
      case "stealth_probe":
      default:
        return {
          ...base,
          tx_id: `PROBE-${Date.now()}-${i}`,
          amount: 1.0 + Math.random(),           // Micro card testing
          merchant_category: "digital_goods",
          device_id: "TEST-RIG",
          city: "Unknown"
        };
    }
  };

  const handleScenario = useCallback(async (key) => {
    if (scenarioRunning) return;
    setScenarioRunning(true);
    setAttackMode(key.toUpperCase());
    
    try {
      const burstCount = 5 + Math.floor(Math.random() * 6); // 5 to 10
      for (let i = 0; i < burstCount; i++) {
        const payload = buildScenarioTx(key, i);
        
        // Single isolated run so loop continues linearly
        try {
          const res = await submitTransaction(payload);
          const syntheticTx = {
            tx_id:             res.transaction_id || res.tx_id,
            user_id:           payload.user_id,
            merchant_id:       payload.merchant_id,
            amount:            payload.amount,
            city:              payload.city,
            channel:           payload.channel,
            device_timezone:   payload.device_timezone,
            risk_score:        Math.round((res.risk_score ?? 0) * 100),
            decision:          res.decision,
            chain_state:       res.chain_state ?? "CLEAR",
            flags:             res.reasons ?? [],
            reasons:           res.reasons ?? [],
            component_scores:  res.component_scores ?? {},
            ml_score:          res.ml_score ?? res.fraud_probability ?? 0,
            rule_score:        res.rule_score ?? 0,
            final_score:       res.final_score ?? 0,
            fraud_probability: res.fraud_probability ?? 0,
            ml_risk_score:     res.ml_risk_score ?? 0,
            ml_explanations:   res.ml_explanations ?? [],
            severity:          res.severity ?? "LOW",
            timestamp:         payload.timestamp,
          };
          
          setTransactions(prev => [syntheticTx, ...prev.filter(t => t.tx_id !== syntheticTx.tx_id)]);
          setSelectedTx(syntheticTx);
          setLastTx(syntheticTx);
          
          // Audio feedback based on risk
          if (res.decision === "BLOCK" || res.decision === "MFA_HOLD") {
            playAlert();
          } else {
            playTick();
          }
        } catch (err) {
          console.error("Burst transaction failed", err);
        }
        
        // Delay 200 - 400 ms realistically
        await new Promise(r => setTimeout(r, 200 + Math.random() * 200));
      }
      
      // Re-fetch backend state at end guarantees DB match
      await fetchTransactions();
    } catch (e) {
      console.error("Scenario burst outer fail:", e);
    } finally {
      setScenarioRunning(false);
      setTimeout(() => setAttackMode(null), 1500);
    }
  }, [fetchTransactions, scenarioRunning]);

  const handleMFAVerify = useCallback(async (otp, purpose) => {
    if (!mfaTx) return;
    try {
      await apiFetch("/mfa/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tx_id: mfaTx.tx_id, otp_code: otp, purpose }),
      });
      setMfaOpen(false);
      setMfaTx(null);
      await fetchTransactions();
    } catch (e) { console.error("MFA verify:", e); }
  }, [mfaTx, fetchTransactions]);

  // ── Handler: TransactionWindow result ─────────────────────────────────────
  const handleTransactionWindowResult = useCallback((result) => {
    // Optimistically prepend the new transaction to the live feed
    const syntheticTx = {
      tx_id:             result.tx_id,
      user_id:           result._payload?.user_id ?? "—",
      merchant_id:       result._payload?.merchant_id ?? "—",
      amount:            result._payload?.amount ?? 0,
      city:              result._payload?.city ?? "—",
      channel:           result._payload?.channel ?? "—",
      device_timezone:   result._payload?.device_timezone ?? "—",
      risk_score:        Math.round((result.risk_score ?? 0) * 100),
      decision:          result.decision,
      chain_state:       result.chain_state ?? "CLEAR",
      flags:             result.reasons ?? [],
      reasons:           result.reasons ?? [],
      component_scores:  result.component_scores ?? {},
      ml_score:          result.ml_score ?? result.fraud_probability ?? 0,
      rule_score:        result.rule_score ?? 0,
      final_score:       result.final_score ?? 0,
      fraud_probability: result.fraud_probability ?? 0,
      ml_risk_score:     result.ml_risk_score ?? 0,
      ml_explanations:   result.ml_explanations ?? [],
      severity:          result.severity ?? "LOW",
      timestamp:         new Date().toISOString(),
      _simType:          result._simType // Forward custom types
    };
    setTransactions((prev) => [syntheticTx, ...prev.filter((t) => t.tx_id !== syntheticTx.tx_id)]);
    setSelectedTx(syntheticTx);
    setLastTx(syntheticTx);
  }, []);

  // ── Derived data ──────────────────────────────────────────────────────────
  const tx = selectedTx ?? transactions[0] ?? null;

  // Use real component_scores from API — NO random math
  const pipelineScores = tx
    ? (() => {
        // component_scores from backend use keys like device_score, behavioral_score etc.
        const cs = tx.component_scores ?? {};
        return {
          amount_score:     cs.amount_score    ?? cs.amount    ?? 0,
          location_score:   cs.location_score  ?? cs.location  ?? 0,
          device_score:     cs.device_score    ?? cs.device    ?? 0,
          behavioral_score: cs.behavioral_score ?? cs.behavioral ?? 0,
          ml_score:         cs.ml_score        ?? cs.ml        ?? 0,
          chain_score:      cs.chain_score     ?? cs.chain     ?? 0,
        };
      })()
    : null;

  const totalTx    = transactions.length;
  const blockedTx  = transactions.filter((t) => t.decision === "BLOCK").length;
  const heldTx     = transactions.filter((t) => t.decision === "MFA_HOLD").length;
  const detectRate = totalTx > 0 ? (((blockedTx + heldTx) / totalTx) * 100).toFixed(1) : "0.0";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className={`min-h-screen transition-all duration-700 text-gray-100 ${
      attackMode 
        ? "bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-red-950/40 via-gray-950 to-black shadow-[inset_0_0_150px_rgba(220,38,38,0.15)]" 
        : "bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-gray-950 to-black"
    }`} style={{ fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ═══════════════════════════ HEADER ══════════════════════════════════ */}
      <header className="sticky top-0 z-50 border-b border-gray-800/80 bg-gray-950/80 backdrop-blur-xl shadow-[0_10px_30px_-10px_rgba(0,0,0,0.5)]">
        <div className="max-w-screen-2xl mx-auto px-6 h-14 flex items-center gap-4">
          {/* Brand */}
          <div className="flex items-center gap-3 mr-4">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-red-600 to-indigo-600 flex items-center justify-center text-sm font-black shadow-lg shadow-red-900/30">
              FS
            </div>
            <div className="flex flex-col leading-none">
              <span className="text-sm font-black text-white tracking-tight">FraudSense</span>
              <span className="text-[9px] text-gray-500 uppercase tracking-widest font-semibold">Intelligence Platform</span>
            </div>
          </div>

          {/* Status */}
          {attackMode ? (
            <motion.div 
               initial={{ scale: 0.9, opacity: 0 }}
               animate={{ scale: 1, opacity: 1 }}
               className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-900/40 border border-red-500/60 shadow-[0_0_20px_rgba(239,68,68,0.6)]"
            >
              <motion.span 
                animate={{ opacity: [0.1, 1, 0.1], scale: [1, 1.4, 1] }}
                transition={{ repeat: Infinity, duration: 0.4 }}
                className="h-2 w-2 rounded-full bg-red-500 shadow-[0_0_10px_#ef4444]" 
              />
              <span className="text-[10px] font-black text-red-100 uppercase tracking-wider">ATTACK MODE ACTIVE: {attackMode}</span>
            </motion.div>
          ) : (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-900/20 border border-emerald-700/40 transition-all duration-300">
              <motion.span 
                animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="h-1.5 w-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" 
              />
              <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">SYSTEM ACTIVE</span>
            </div>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* Live stats */}
          <div className="hidden sm:flex items-center gap-5 text-xs text-gray-400">
            <StatPill label="Transactions" value={totalTx} />
            <StatPill label="Detection Rate" value={`${detectRate}%`} highlight={parseFloat(detectRate) > 30} />
            <StatPill label="Blocked" value={blockedTx} highlight={blockedTx > 0} />
            <StatPill label="MFA Hold" value={heldTx} warn={heldTx > 0} />
          </div>

          {/* Coordinated attack pill */}
          {coordinatedAlert?.coordinated && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-red-900/30 border border-red-700/50 animate-pulse">
              <span className="h-1.5 w-1.5 rounded-full bg-red-400 animate-ping" />
              <span className="text-[10px] font-bold text-red-400 uppercase tracking-wider">⚡ ATTACK DETECTED</span>
            </div>
          )}
        </div>
      </header>

      {/* ═══════════════════════ SCENARIO CONTROLS ═══════════════════════════ */}
      <div className="sticky top-14 z-40 border-b border-gray-800/60 bg-gray-950/90 backdrop-blur">
        <div className="max-w-screen-2xl mx-auto px-6 py-2 flex items-center gap-3 flex-wrap">
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mr-1">Simulate:</span>
          {[
            { key: "ato",         label: "ATO Attack",        color: "bg-red-700 hover:bg-red-600",    icon: "🎯" },
            { key: "coordinated", label: "Coordinated",       color: "bg-orange-700 hover:bg-orange-600", icon: "👥" },
            { key: "mimicry",     label: "Mimicry",           color: "bg-purple-700 hover:bg-purple-600", icon: "🎭" },
            { key: "scam",        label: "APP Scam",          color: "bg-amber-700 hover:bg-amber-600",  icon: "💸" },
            { key: "stealth_probe",label: "Stealth Probe",    color: "bg-blue-800 hover:bg-blue-700",   icon: "🔍" },
          ].map(({ key, label, color, icon }) => (
            <motion.button
              whileHover={{ scale: 1.03, filter: "brightness(1.2)" }}
              whileTap={{ scale: 0.95 }}
              key={key}
              onClick={() => handleScenario(key)}
              disabled={scenarioRunning}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold text-white transition-all duration-300
                disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl hover:shadow-${color.replace('bg-', '').split('-')[0]}-500/30 ${color} ${scenarioRunning ? "animate-pulse" : ""}`}
            >
              <span className="drop-shadow-md text-sm">{icon}</span>
              <span className="drop-shadow-md">{scenarioRunning ? "Running…" : label}</span>
            </motion.button>
          ))}

          {/* Tab toggle */}
          <div className="ml-auto flex items-center gap-1 bg-gray-900 rounded-lg p-1 border border-gray-800">
            {[["command", "⬡ Command Center"], ["feed", "📋 Transaction Feed"]].map(([id, label]) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`px-3 py-1 rounded-md text-[11px] font-semibold transition-all ${
                  activeTab === id
                    ? "bg-indigo-600 text-white shadow"
                    : "text-gray-500 hover:text-gray-300"
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ═══════════════════════════ METRICS STRIP ═══════════════════════════ */}
      <div className="max-w-screen-2xl mx-auto px-6 py-3">
        <MetricsPanel metrics={metrics} transactions={transactions} />
      </div>

      {/* ═══════════════════ MAIN COMMAND CENTER LAYOUT ══════════════════════ */}
      <main className="max-w-screen-2xl mx-auto px-6 pb-10">

        {/* TAB: COMMAND CENTER */}
        {activeTab === "command" && (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">

            {/* ── LEFT: Transaction Window + Live Feed ───────────────────── */}
            <div className="xl:col-span-3 flex flex-col gap-4">
              {/* Transaction Input Window */}
              <Card title="Submit Transaction" icon="💳">
                <TransactionWindow onResult={handleTransactionWindowResult} />
              </Card>

              {/* Stress Test Simulation */}
              <Card title="Live Stress Test" icon="🧪">
                <TransactionSimulator onResult={handleTransactionWindowResult} />
              </Card>

              {/* Live Feed */}
              <Card title="Live Feed" icon="🔴" badge={`${transactions.length} TXN`}>
                <div className="overflow-y-auto overflow-x-hidden" style={{ maxHeight: "420px" }}>
                  <AnimatePresence initial={false}>
                    {transactions.length === 0 ? (
                      <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="flex flex-col items-center justify-center py-16 text-gray-600 gap-3"
                      >
                        <span className="text-3xl">📭</span>
                        <p className="text-sm">No transactions yet</p>
                        <p className="text-xs text-gray-700">Submit a transaction or run a scenario</p>
                      </motion.div>
                    ) : (
                      transactions.map((t) => (
                        <motion.div
                          key={t.tx_id}
                          initial={{ opacity: 0, x: -20, height: 0 }}
                          animate={{ opacity: 1, x: 0, height: "auto" }}
                          exit={{ opacity: 0, x: 20, height: 0 }}
                          transition={{ duration: 0.3 }}
                        >
                          <TxRow
                            tx={t}
                            isSelected={selectedTx?.tx_id === t.tx_id}
                            onClick={() => handleSelect(t)}
                          />
                        </motion.div>
                      ))
                    )}
                  </AnimatePresence>
                </div>
              </Card>
            </div>

            {/* ── CENTER: Risk Panel + Trigger Insights + Chain ─────────── */}
            <div className="xl:col-span-5 flex flex-col gap-4">
              {/* Hero Risk Panel */}
              <Card icon="🛡" title="Risk Assessment" noPad>
                <RiskPanel tx={tx} />
              </Card>

              {/* AI Decision Engine */}
              <Card icon="🧠" title="AI Decision Engine" noPad>
                <div className="p-4">
                  <AIDecisionPanel tx={tx} />
                </div>
              </Card>

              {/* Chain Sequence */}
              <Card>
                <ChainView
                  chainState={tx?.chain_state ?? "CLEAN"}
                  eventLog={tx?.event_log ?? null}
                />
              </Card>
            </div>

            {/* ── RIGHT: Graph + Pipeline + Adaptive + Case File ─────────── */}
            <div className="xl:col-span-4 flex flex-col gap-4">
              {/* Graph Intelligence */}
              <Card>
                <GraphAlert coordinatedAttack={coordinatedAlert} />
              </Card>

              {/* Pipeline Breakdown */}
              <Card>
                <PipelineBreakdown
                  pipelineScores={pipelineScores}
                  finalScore={tx?.risk_score ?? null}
                />
              </Card>

              {/* Adaptive Learning */}
              <Card>
                <AdaptivePanel
                  dynamicWeights={dynamicWeights}
                  weightHistory={weightHistory}
                />
              </Card>

              {/* Case File */}
              <Card>
                <CaseFilePanel data={lastTx} selectedCase={selectedCase} txId={lastTx?.tx_id ?? tx?.tx_id ?? null} />
              </Card>
            </div>
          </div>
        )}

        {/* TAB: TRANSACTION FEED (full-width detailed table) */}
        {activeTab === "feed" && (
          <div className="rounded-2xl border border-gray-800 bg-gray-900/50 overflow-hidden">
            <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-800">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
              <h2 className="text-sm font-bold text-gray-300 uppercase tracking-widest">All Transactions</h2>
              <span className="ml-auto text-xs text-gray-600">{transactions.length} records</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 uppercase tracking-wider">
                    {["Tx ID", "User", "Amount", "Merchant", "Risk", "Decision", "Chain", "Timestamp"].map((h) => (
                      <th key={h} className="px-4 py-3 text-left font-bold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((t) => {
                    const dc = decisionColor(t.decision);
                    const riskColor = (t.risk_score ?? 0) >= 66 ? "text-red-400" : (t.risk_score ?? 0) >= 36 ? "text-amber-400" : "text-emerald-400";
                    return (
                      <tr
                        key={t.tx_id}
                        onClick={() => { handleSelect(t); setActiveTab("command"); }}
                        className="border-b border-gray-800/60 hover:bg-gray-800/50 cursor-pointer transition-colors"
                      >
                        <td className="px-4 py-3 font-mono text-gray-300">{t.tx_id}</td>
                        <td className="px-4 py-3 text-gray-400">{t.user_id}</td>
                        <td className="px-4 py-3 font-semibold text-gray-200">
                          {new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(t.amount)}
                        </td>
                        <td className="px-4 py-3 text-gray-400">{t.merchant_id}</td>
                        <td className={`px-4 py-3 font-black tabular-nums ${riskColor}`}>{t.risk_score ?? "—"}</td>
                        <td className="px-4 py-3">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${dc}`}>
                            {t.decision}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-gray-500">{t.chain_state || "—"}</td>
                        <td className="px-4 py-3 text-gray-600 font-mono">
                          {t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {transactions.length === 0 && (
                <div className="text-center py-24 text-gray-700">
                  <p className="text-3xl mb-3">📭</p>
                  <p>No transactions yet. Run a scenario!</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* ══════════════════════════ MFA MODAL ════════════════════════════════ */}
      {mfaOpen && mfaTx && (
        <MFAModal
          transaction={mfaTx}
          onVerify={handleMFAVerify}
          onCancel={() => { setMfaOpen(false); setMfaTx(null); }}
        />
      )}
    </div>
  );
}

// ── Shared Card wrapper ────────────────────────────────────────────────────────
function Card({ children, title, icon, badge, noPad = false }) {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      whileHover={{ y: -2, boxShadow: "0 10px 40px -10px rgba(0,0,0,0.5)" }}
      transition={{ type: "spring", stiffness: 300, damping: 20, mass: 1 }}
      className="rounded-2xl border border-gray-800/70 bg-gray-900/60 backdrop-blur overflow-hidden group transition-colors duration-300 relative"
    >
      {(title || icon || badge) && (
        <div className="flex items-center gap-2 px-5 py-3 border-b border-gray-800/60 relative overflow-hidden bg-gradient-to-r from-gray-900/40 to-transparent">
          {/* Subtle glow effect revealed on hover */}
          <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

          {icon && <span className="text-sm relative z-10">{icon}</span>}
          {title && <h3 className="text-[10px] font-bold text-gray-500 uppercase tracking-widest relative z-10">{title}</h3>}
          {badge && (
            <span className="ml-auto text-[10px] font-bold text-gray-600 bg-gray-800 px-2 py-0.5 rounded-full border border-gray-700 relative z-10">
              {badge}
            </span>
          )}
        </div>
      )}
      <div className={`${noPad ? "" : "p-4"} relative z-10`}>{children}</div>
    </motion.div>
  );
}

// ── Header stat pill ──────────────────────────────────────────────────────────
function StatPill({ label, value, highlight, warn }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-gray-600">{label}:</span>
      <span className={`font-bold tabular-nums ${
        highlight ? "text-red-400" : warn ? "text-amber-400" : "text-gray-300"
      }`}>{value}</span>
    </div>
  );
}