/**
 * TransactionSimulator.jsx
 * Uses the backend's /api/simulate endpoint for realistic diverse transactions.
 * Distribution: 40% SAFE · 35% MFA · 25% BLOCK
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import { motion } from "framer-motion";

const API_BASE = "/api";   // Vite proxy → localhost:8000

const PROFILES = [
  { label: "Random",  value: "random",  color: "bg-indigo-700 hover:bg-indigo-600",   icon: "🎲" },
  { label: "Safe",    value: "safe",    color: "bg-emerald-700 hover:bg-emerald-600",  icon: "✅" },
  { label: "MFA",     value: "mfa",     color: "bg-amber-700 hover:bg-amber-600",      icon: "⚠️" },
  { label: "Block",   value: "block",   color: "bg-red-700 hover:bg-red-600",          icon: "🛑" },
];

function DecisionBadge({ decision }) {
  const cfg = {
    BLOCK:    { cls: "bg-red-900/40 text-red-400 border-red-700/50",     label: "BLOCK"   },
    MFA_HOLD: { cls: "bg-amber-900/40 text-amber-400 border-amber-700/50", label: "MFA HOLD" },
    APPROVE:  { cls: "bg-emerald-900/40 text-emerald-400 border-emerald-700/50", label: "APPROVE" },
  }[decision] ?? { cls: "bg-gray-800 text-gray-400 border-gray-700", label: decision };
  return (
    <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full border uppercase tracking-wider ${cfg.cls}`}>
      {cfg.label}
    </span>
  );
}

export default function TransactionSimulator({ onResult }) {
  const [isActive,   setIsActive]   = useState(false);
  const [profile,    setProfile]    = useState("random");
  const [count,      setCount]      = useState(0);
  const [lastResult, setLastResult] = useState(null);
  const [stats,      setStats]      = useState({ APPROVE: 0, MFA_HOLD: 0, BLOCK: 0 });
  const [loading,    setLoading]    = useState(false);
  const timerRef = useRef(null);

  // Single simulation via /api/simulate
  const simulate = useCallback(async (prof = profile) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/simulate?profile=${prof}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // Normalize for App.jsx's handleTransactionWindowResult contract
      const normalized = {
        tx_id:            data.tx_id,
        transaction_id:   data.tx_id,
        risk_score:       data.risk_score,
        decision:         data.decision,
        reasons:          data.reasons ?? [],
        component_scores: data.component_scores ?? {},
        chain_state:      data.chain_state ?? "CLEAR",
        ml_risk_score:    data.ml_risk_score ?? 0,
        fraud_probability: data.fraud_probability ?? 0,
        ml_explanations:  data.ml_explanations ?? [],
        ml_score:         data.ml_score ?? data.fraud_probability ?? 0,
        rule_score:       data.rule_score ?? 0,
        final_score:      data.final_score ?? 0,
        severity:         data.severity ?? "LOW",
        _simType:         prof === "safe" ? "NORMAL" : prof === "block" ? "FRAUD" : "SUSPICIOUS",
        _payload: {
          user_id:     data.user_id ?? "sim_user",
          merchant_id: data.merchant_id ?? "sim_merch",
          amount:      data.amount ?? 0,
          city:        data.city ?? "—",
          channel:     "simulator",
          device_timezone: "Asia/Kolkata",
        }
      };

      setLastResult(normalized);
      setCount(c => c + 1);
      setStats(prev => ({ ...prev, [data.decision]: (prev[data.decision] ?? 0) + 1 }));
      if (onResult) onResult(normalized);
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      setLoading(false);
    }
  }, [profile, onResult]);

  // Auto-mode interval
  useEffect(() => {
    if (isActive) {
      timerRef.current = setInterval(() => simulate(), 2500);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [isActive, simulate]);

  const totalStats = Object.values(stats).reduce((a, b) => a + b, 0);

  return (
    <div className="flex flex-col gap-3 p-4 bg-gray-900/40 border border-indigo-500/20 rounded-xl">

      {/* Profile selector */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[9px] font-bold text-gray-600 uppercase tracking-widest mr-1">Profile:</span>
        {PROFILES.map(p => (
          <button
            key={p.value}
            onClick={() => setProfile(p.value)}
            className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-bold text-white transition-all duration-200 ${p.color}
              ${profile === p.value ? "ring-2 ring-white/30 scale-105" : "opacity-60 hover:opacity-90"}`}
          >
            <span>{p.icon}</span>{p.label}
          </button>
        ))}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        {/* Manual single fire */}
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={() => simulate(profile)}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-indigo-700 hover:bg-indigo-600 rounded-lg text-[11px] font-bold text-white transition-all disabled:opacity-50"
        >
          {loading ? <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : "⚡"}
          Fire One
        </motion.button>

        {/* Auto toggle */}
        <button
          onClick={() => setIsActive(v => !v)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold text-white transition-all ${
            isActive ? "bg-red-700 hover:bg-red-600 animate-pulse" : "bg-gray-700 hover:bg-gray-600"
          }`}
        >
          {isActive ? "⏹ Stop" : "▶ Auto"}
        </button>

        <span className="ml-auto text-[10px] text-gray-600 font-mono tabular-nums">{count} fired</span>
      </div>

      {/* Stats bar */}
      {totalStats > 0 && (
        <div className="flex gap-2 text-[10px] font-bold">
          <span className="text-emerald-400">✓ {stats.APPROVE ?? 0}</span>
          <span className="text-amber-400">⚠ {stats.MFA_HOLD ?? 0}</span>
          <span className="text-red-400">✗ {stats.BLOCK ?? 0}</span>
          <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden flex ml-1 self-center">
            {totalStats > 0 && <>
              <div className="bg-emerald-500 h-full" style={{ width: `${(stats.APPROVE ?? 0) / totalStats * 100}%` }} />
              <div className="bg-amber-500 h-full" style={{ width: `${(stats.MFA_HOLD ?? 0) / totalStats * 100}%` }} />
              <div className="bg-red-500 h-full" style={{ width: `${(stats.BLOCK ?? 0) / totalStats * 100}%` }} />
            </>}
          </div>
        </div>
      )}

      {/* Last result */}
      {lastResult && (
        <div className="flex items-center gap-2 pt-1 border-t border-gray-800/60">
          <span className="text-[9px] text-gray-600 font-mono truncate flex-1">{lastResult.tx_id}</span>
          <DecisionBadge decision={lastResult.decision} />
          <span className="text-[9px] text-gray-600 tabular-nums">
            {Math.round((lastResult.risk_score ?? 0) * 100)}/100
          </span>
        </div>
      )}
    </div>
  );
}