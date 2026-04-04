/**
 * TransactionWindow.jsx
 * Neon fintech-style transaction evaluation panel
 * Hackathon-demo ready: 4 inputs → real-time fraud decision
 * Connects to: POST /api/transaction
 */

import React, { useState, useCallback, useRef, useEffect } from "react";
import { motion, animate, useMotionValue, useTransform } from "framer-motion";


// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = "/api";   // Vite proxy → localhost:8000

const LOCATIONS = [
  "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
  "New York", "Chicago", "Seattle", "Miami", "Lagos",
  "Dubai", "London", "Singapore", "Unknown",
];

const CITY_TZ = {
  Mumbai: "Asia/Kolkata", Delhi: "Asia/Kolkata", Bangalore: "Asia/Kolkata",
  Chennai: "Asia/Kolkata", Kolkata: "Asia/Kolkata",
  "New York": "America/New_York", Chicago: "America/Chicago",
  Seattle: "America/Los_Angeles", Miami: "America/New_York",
  Lagos: "Africa/Lagos", Dubai: "Asia/Dubai",
  London: "Europe/London", Singapore: "Asia/Singapore",
  Unknown: "UTC",
};

const genTxId = () =>
  `TX-${Date.now().toString(36).toUpperCase()}-${Math.random().toString(36).slice(2,6).toUpperCase()}`;

// ─────────────────────────────────────────────────────────────────────────────
// Decision config
// ─────────────────────────────────────────────────────────────────────────────

const DECISION_CONFIG = {
  BLOCK: {
    label:     "BLOCKED",
    sublabel:  "Transaction Prevented",
    icon:      "🚫",
    neon:      "#ef4444",
    neonDim:   "#7f1d1d",
    textClass: "text-red-300",
    bgClass:   "bg-red-500/10",
    borderClass: "border-red-500/40",
    glowClass: "shadow-red-500/20",
    barColor:  "#ef4444",
  },
  MFA_HOLD: {
    label:     "MFA REQUIRED",
    sublabel:  "Verification Needed",
    icon:      "⚠️",
    neon:      "#f59e0b",
    neonDim:   "#78350f",
    textClass: "text-amber-300",
    bgClass:   "bg-amber-500/10",
    borderClass: "border-amber-500/40",
    glowClass: "shadow-amber-500/20",
    barColor:  "#f59e0b",
  },
  APPROVE: {
    label:     "APPROVED",
    sublabel:  "Transaction Cleared",
    icon:      "✅",
    neon:      "#10b981",
    neonDim:   "#064e3b",
    textClass: "text-emerald-300",
    bgClass:   "bg-emerald-500/10",
    borderClass: "border-emerald-500/40",
    glowClass: "shadow-emerald-500/20",
    barColor:  "#10b981",
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Animated Risk Bar
// ─────────────────────────────────────────────────────────────────────────────

function RiskBar({ score01, color }) {
  const pct = Math.round((score01 ?? 0) * 100);
  const count = useMotionValue(0);
  const rounded = useTransform(count, Math.round);

  useEffect(() => {
    const controls = animate(count, pct, { duration: 0.7, ease: "easeOut" });
    return controls.stop;
  }, [pct]);

  const segmentColor =
    pct >= 70 ? "#ef4444" :
    pct >= 40 ? "#f59e0b" :
               "#10b981";

  return (
    <div className="flex flex-col gap-2">
      {/* Label row */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-bold text-gray-500 uppercase tracking-widest">
          Risk Score
        </span>
        <motion.span
          className="text-2xl font-black tabular-nums transition-colors duration-300 flex items-baseline"
          style={{ color: segmentColor, textShadow: `0 0 20px ${segmentColor}88` }}
        >
          <motion.span>{rounded}</motion.span>
          <span className="text-sm text-gray-600 font-normal ml-0.5">/100</span>
        </motion.span>
      </div>

      {/* Track */}
      <div className="relative h-3 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
        {/* Segment markers */}
        {[40, 70].map((marker) => (
          <div
            key={marker}
            className="absolute top-0 bottom-0 w-px bg-gray-700/60 z-10"
            style={{ left: `${marker}%` }}
          />
        ))}
        {/* Fill */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ 
            width: `${pct}%`,
            boxShadow: pct > 0 ? `0 0 16px ${segmentColor}66, 0 0 4px ${segmentColor}` : "none"
          }}
          transition={{ duration: 0.7, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${segmentColor}55, ${segmentColor})` }}
        />
      </div>

      {/* Threshold labels */}
      <div className="relative flex text-[9px] font-bold uppercase tracking-wider">
        <span className="text-emerald-600" style={{ width: "40%" }}>Approve</span>
        <span className="text-amber-600 text-center" style={{ width: "30%" }}>MFA</span>
        <span className="text-red-600 text-right flex-1">Block</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Spinner
// ─────────────────────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <div className="flex flex-col items-center gap-4 py-8">
      {/* Outer ring */}
      <div className="relative w-16 h-16">
        <div
          className="absolute inset-0 rounded-full border-2 border-transparent animate-spin"
          style={{
            borderTopColor: "#6366f1",
            borderRightColor: "#6366f155",
            animationDuration: "0.8s",
          }}
        />
        <div
          className="absolute inset-2 rounded-full border-2 border-transparent animate-spin"
          style={{
            borderTopColor: "#8b5cf6",
            borderLeftColor: "#8b5cf655",
            animationDuration: "1.2s",
            animationDirection: "reverse",
          }}
        />
        <div className="absolute inset-0 flex items-center justify-center text-lg">
          🔍
        </div>
      </div>
      <div className="flex flex-col items-center gap-1">
        <p className="text-sm font-bold text-indigo-300">Evaluating Transaction</p>
        <p className="text-[11px] text-gray-600">Running fraud detection pipeline…</p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Input + Label helpers
// ─────────────────────────────────────────────────────────────────────────────

function Field({ label, icon, children }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-1.5 text-[10px] font-bold text-gray-500 uppercase tracking-widest">
        <span>{icon}</span>
        {label}
      </label>
      {children}
    </div>
  );
}

const inputCls =
  "w-full bg-gray-950 border border-gray-800 rounded-lg px-3 py-2.5 text-sm text-gray-100 " +
  "placeholder-gray-700 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 " +
  "transition-all duration-200 hover:border-gray-700";

const selectCls =
  "w-full bg-gray-950 border border-gray-800 rounded-lg px-3 py-2.5 text-sm text-gray-100 " +
  "focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/30 " +
  "transition-all duration-200 hover:border-gray-700 cursor-pointer";

// ─────────────────────────────────────────────────────────────────────────────
// Main TransactionWindow
// ─────────────────────────────────────────────────────────────────────────────

export default function TransactionWindow({ onResult }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState(null);

  // Core 4 required fields + hidden required fields
  const [amount,   setAmount]   = useState("");
  const [location, setLocation] = useState("Mumbai");
  const [deviceId, setDeviceId] = useState("DEV-DEMO-01");
  const [merchant, setMerchant] = useState("MERCH-001");

  // Advanced (collapsed by default)
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [userId,    setUserId]    = useState("user_demo_01");
  const [txType,    setTxType]    = useState("PAYMENT");
  const [category,  setCategory]  = useState("retail");
  const [oldBalance, setOldBalance] = useState("10000");

  const resultRef = useRef(null);

  // Auto-scroll to result
  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [result]);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!amount) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const amt   = parseFloat(amount) || 0;
    const oldBal = parseFloat(oldBalance) || 0;
    const newBal = Math.max(0, oldBal - amt);
    const tz    = CITY_TZ[location] || "UTC";

    const payload = {
      tx_id:             genTxId(),
      user_id:           userId,
      amount:            amt,
      merchant_id:       merchant,
      merchant_category: category,
      device_id:         deviceId,
      city:              location,
      device_timezone:   tz,
      timestamp:         new Date().toISOString(),
      channel:           "web",
      oldbalanceOrg:     oldBal,
      newbalanceOrig:    newBal,
      oldbalanceDest:    0,
      newbalanceDest:    amt,
      tx_type:           txType,
    };

    try {
      const t0  = performance.now();
      const res = await fetch(`${API_BASE}/transaction`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      const data    = await res.json();
      const latency = Math.round(performance.now() - t0);
      const enriched = { ...data, _latency_ms: latency, _payload: payload };

      setResult(enriched);
      if (onResult) onResult(enriched);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [amount, location, deviceId, merchant, userId, txType, category, oldBalance, onResult]);

  const cfg = result ? (DECISION_CONFIG[result.decision] ?? DECISION_CONFIG.APPROVE) : null;
  const riskPct = result ? Math.round((result.risk_score ?? 0) * 100) : 0;

  return (
    <div
      className="flex flex-col gap-0 rounded-2xl overflow-hidden"
      style={{
        background: "linear-gradient(145deg, #0d0d14 0%, #0a0a12 100%)",
        border: "1px solid rgba(99,102,241,0.15)",
        boxShadow: "0 0 40px rgba(99,102,241,0.05), inset 0 1px 0 rgba(255,255,255,0.03)",
      }}
    >
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div
        className="flex items-center gap-3 px-5 py-4"
        style={{
          borderBottom: "1px solid rgba(99,102,241,0.12)",
          background: "linear-gradient(90deg, rgba(99,102,241,0.08) 0%, transparent 100%)",
        }}
      >
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-shrink-0"
          style={{
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            boxShadow: "0 0 16px rgba(99,102,241,0.4)",
          }}
        >
          💳
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-black text-white tracking-tight">Process Transaction</p>
          <p className="text-[10px] text-gray-600 font-medium">Real-time fraud evaluation</p>
        </div>
        {/* Live indicator */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-900/20 border border-emerald-500/20">
          <motion.span 
            animate={{ opacity: [0.3, 1, 0.3], scale: [1, 1.2, 1] }}
            transition={{ repeat: Infinity, duration: 2 }}
            className="w-1.5 h-1.5 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" 
          />
          <span className="text-[9px] font-bold text-emerald-400 uppercase tracking-widest">Live</span>
        </div>
      </div>

      {/* ── Form body ────────────────────────────────────────────────────── */}
      <div className="p-5 flex flex-col gap-4">
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">

          {/* ── Core 4 fields ─────────────────────────────────────────── */}
          {/* Amount */}
          <Field label="Amount" icon="💰">
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-600 text-sm font-bold">₹</span>
              <input
                id="tw-amount"
                type="number"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder="0.00"
                min="0"
                step="0.01"
                className={inputCls + " pl-7"}
              />
            </div>
            {/* Quick amount presets */}
            <div className="flex gap-1.5 flex-wrap">
              {[1000, 5000, 25000, 100000].map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setAmount(String(preset))}
                  className={`px-2 py-0.5 rounded-md text-[10px] font-bold border transition-all duration-150
                    ${amount === String(preset)
                      ? "bg-indigo-600/30 border-indigo-500/60 text-indigo-300"
                      : "bg-gray-900 border-gray-800 text-gray-600 hover:border-gray-700 hover:text-gray-400"
                    }`}
                >
                  ₹{preset >= 1000 ? `${preset/1000}k` : preset}
                </button>
              ))}
            </div>
          </Field>

          {/* Location + Device row */}
          <div className="grid grid-cols-2 gap-3">
            <Field label="Location" icon="📍">
              <select
                id="tw-location"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className={selectCls}
              >
                {LOCATIONS.map((loc) => (
                  <option key={loc} value={loc}>{loc}</option>
                ))}
              </select>
            </Field>
            <Field label="Device ID" icon="📱">
              <input
                id="tw-device"
                type="text"
                value={deviceId}
                onChange={(e) => setDeviceId(e.target.value)}
                placeholder="DEV-001"
                className={inputCls}
              />
            </Field>
          </div>

          {/* Merchant */}
          <Field label="Merchant" icon="🏪">
            <div className="grid grid-cols-2 gap-3">
              <input
                id="tw-merchant"
                type="text"
                value={merchant}
                onChange={(e) => setMerchant(e.target.value)}
                placeholder="MERCH-001"
                className={inputCls}
              />
              <select
                id="tw-category"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className={selectCls}
              >
                {[
                  "retail", "groceries", "restaurants", "electronics",
                  "cryptocurrency", "wire_transfer", "gambling", "forex",
                  "healthcare", "travel", "digital_goods",
                ].map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>
          </Field>

          {/* ── Advanced toggle ─────────────────────────────────────── */}
          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            className="flex items-center gap-1.5 text-[10px] font-bold text-gray-600 hover:text-gray-400 transition-colors self-start"
          >
            <span className={`transition-transform duration-200 ${showAdvanced ? "rotate-90" : ""}`}>›</span>
            {showAdvanced ? "Hide" : "Show"} advanced options
          </button>

          {showAdvanced && (
            <div className="grid grid-cols-2 gap-3 animate-[fadeIn_0.2s_ease]">
              <Field label="User ID" icon="👤">
                <input
                  id="tw-userid"
                  type="text"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className={inputCls}
                />
              </Field>
              <Field label="Tx Type" icon="🔄">
                <select
                  id="tw-txtype"
                  value={txType}
                  onChange={(e) => setTxType(e.target.value)}
                  className={selectCls}
                >
                  {["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"].map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </Field>
              <Field label="Balance Before" icon="🏦">
                <input
                  id="tw-oldbal"
                  type="number"
                  value={oldBalance}
                  onChange={(e) => setOldBalance(e.target.value)}
                  className={inputCls}
                />
              </Field>
            </div>
          )}

          {/* ── Submit button ───────────────────────────────────────── */}
          <motion.button
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.98 }}
            id="tw-submit"
            type="submit"
            disabled={loading || !amount}
            className="relative w-full flex items-center justify-center gap-2.5 py-3.5 rounded-xl
              font-bold text-sm text-white transition-opacity duration-200
              disabled:opacity-40 disabled:cursor-not-allowed overflow-hidden"
            style={{
              background: loading
                ? "linear-gradient(135deg, #3730a3, #4c1d95)"
                : "linear-gradient(135deg, #4f46e5, #7c3aed)",
              boxShadow: loading
                ? "none"
                : "0 0 24px rgba(99,102,241,0.35), 0 4px 15px rgba(99,102,241,0.2)",
            }}
          >
            {/* Shimmer effect when idle */}
            {!loading && (
              <span
                className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-300"
                style={{
                  background: "linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 50%)",
                }}
              />
            )}
            {loading ? (
              <>
                <span
                  className="w-4 h-4 rounded-full border-2 border-indigo-300 border-t-transparent animate-spin"
                />
                Processing…
              </>
            ) : (
              <>
                <span>⚡</span>
                Process Transaction
              </>
            )}
          </motion.button>
        </form>

        {/* ── Loading animation ──────────────────────────────────────────── */}
        {loading && <Spinner />}

        {/* ── Error state ───────────────────────────────────────────────── */}
        {error && !loading && (
          <div
            className="rounded-xl p-3.5 flex items-start gap-2.5 text-xs"
            style={{
              background: "rgba(127,29,29,0.15)",
              border: "1px solid rgba(239,68,68,0.25)",
            }}
          >
            <span className="flex-shrink-0 mt-0.5">⚠</span>
            <div>
              <p className="font-bold text-red-400 mb-0.5">Connection Error</p>
              <p className="text-red-500/80">{error}</p>
              <p className="text-gray-700 mt-1">Ensure the backend is running on port 8000</p>
            </div>
          </div>
        )}

        {/* ── Result panel ──────────────────────────────────────────────── */}
        {result && !loading && cfg && (
          <div
            ref={resultRef}
            className="flex flex-col gap-5 rounded-xl p-4 transition-all duration-500"
            style={{
              background: `linear-gradient(145deg, ${cfg.neonDim}22 0%, rgba(10,10,18,0.8) 100%)`,
              border: `1px solid ${cfg.neon}33`,
              boxShadow: `0 0 40px ${cfg.neon}15, inset 0 1px 0 ${cfg.neon}10`,
            }}
          >
            {/* Decision banner */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div
                  className="w-11 h-11 rounded-xl flex items-center justify-center text-xl flex-shrink-0"
                  style={{
                    background: `${cfg.neon}18`,
                    border: `1px solid ${cfg.neon}40`,
                    boxShadow: `0 0 20px ${cfg.neon}20`,
                    animation: result.decision !== "APPROVE" ? "pulse 2s infinite" : "none",
                  }}
                >
                  {cfg.icon}
                </div>
                <div>
                  <p
                    className="text-base font-black tracking-wide"
                    style={{
                      color: cfg.neon,
                      textShadow: `0 0 20px ${cfg.neon}66`,
                    }}
                  >
                    {cfg.label}
                  </p>
                  <p className="text-[10px] font-semibold text-gray-600 uppercase tracking-widest mt-0.5">
                    {cfg.sublabel} · {result._latency_ms}ms
                  </p>
                </div>
              </div>

              {/* Risk score badge */}
              <div className="text-right flex-shrink-0">
                <p
                  className="text-4xl font-black tabular-nums leading-none"
                  style={{
                    color: cfg.neon,
                    textShadow: `0 0 24px ${cfg.neon}88`,
                  }}
                >
                  {riskPct}
                </p>
                <p className="text-[10px] text-gray-700 font-medium mt-1">RISK SCORE</p>
              </div>
            </div>

            {/* Animated risk progress bar */}
            <RiskBar score01={result.risk_score} color={cfg.barColor} />

            {/* Separator */}
            <div style={{ height: "1px", background: `${cfg.neon}18` }} />

            {/* Reasons */}
            {result.reasons?.length > 0 && (
              <div className="flex flex-col gap-2">
                <p className="text-[10px] font-bold uppercase tracking-widest"
                  style={{ color: `${cfg.neon}88` }}>
                  Triggered Signals
                </p>
                <ul className="flex flex-col gap-2">
                  {result.reasons.slice(0, 5).map((reason, i) => (
                    <li
                      key={i}
                      className="flex items-start gap-2.5 text-xs rounded-lg px-3 py-2"
                      style={{
                        background: `${cfg.neon}08`,
                        border: `1px solid ${cfg.neon}15`,
                      }}
                    >
                      <span
                        className="flex-shrink-0 mt-0.5 w-1.5 h-1.5 rounded-full"
                        style={{ background: cfg.neon, marginTop: "5px" }}
                      />
                      <span className="text-gray-400 leading-relaxed">{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* No reasons = clean transaction */}
            {(!result.reasons || result.reasons.length === 0) && (
              <p className="text-xs text-gray-600 flex items-center gap-2">
                <span style={{ color: cfg.neon }}>✓</span>
                No anomalies detected — transaction profile is clean
              </p>
            )}

            {/* Pre-tx block note */}
            {result.pre_tx_blocked && (
              <div
                className="rounded-lg px-3 py-2.5 text-xs font-semibold flex items-center gap-2"
                style={{
                  background: "rgba(127,29,29,0.25)",
                  border: "1px solid rgba(239,68,68,0.3)",
                  color: "#fca5a5",
                }}
              >
                🛑 Transaction intercepted — funds not transferred
              </div>
            )}

            {/* Footer */}
            <div className="flex items-center justify-between">
              <p className="text-[10px] text-gray-700 font-mono tracking-tight">
                {result.tx_id}
              </p>
              <button
                onClick={() => { setResult(null); setError(null); setAmount(""); }}
                className="text-[10px] font-bold text-gray-700 hover:text-gray-400 transition-colors uppercase tracking-widest"
              >
                New Transaction ›
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
