/**
 * RiskPanel.jsx
 * Hero risk gauge with score ring, decision badge, and transaction metadata.
 * Shows ML Score, Rule Score, and Final Score bars.
 */

import React, { useEffect, useRef } from "react";
import { motion } from "framer-motion";

const formatCurrency = (amount) => {
  if (amount == null) return "—";
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(amount);
};

function RiskRing({ score }) {
  const canvasRef = useRef(null);
  const clampedScore = Math.min(100, Math.max(0, score || 0));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const size = canvas.width;
    const cx = size / 2, cy = size / 2;
    const radius = size * 0.42;
    const lineWidth = size * 0.08;
    const startAngle = -Math.PI * 0.75;
    const totalAngle = Math.PI * 1.5;

    ctx.clearRect(0, 0, size, size);

    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + totalAngle);
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();

    // Fill
    const fillAngle = (clampedScore / 100) * totalAngle;
    const color = clampedScore >= 66 ? "#ef4444" : clampedScore >= 36 ? "#f59e0b" : "#10b981";
    const grd = ctx.createLinearGradient(0, 0, size, size);
    grd.addColorStop(0, color + "aa");
    grd.addColorStop(1, color);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + fillAngle);
    ctx.strokeStyle = grd;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [clampedScore]);

  const color = clampedScore >= 66 ? "text-red-400" : clampedScore >= 36 ? "text-amber-400" : "text-emerald-400";

  return (
    <div className="relative inline-flex items-center justify-center">
      <canvas ref={canvasRef} width={180} height={180} />
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-4xl font-black tabular-nums ${color}`}>{clampedScore.toFixed(1)}</span>
        <span className="text-xs text-gray-400 font-semibold tracking-widest uppercase mt-1">Risk Score</span>
      </div>
    </div>
  );
}

function DecisionBadge({ decision }) {
  const cfg = {
    BLOCK: { label: "❌ BLOCK", cls: "bg-red-500/20 text-red-300 border-red-500/50 shadow-red-500/30", pulse: true },
    MFA_HOLD: { label: "⚠️ MFA HOLD", cls: "bg-amber-500/20 text-amber-300 border-amber-500/50 shadow-amber-500/30", pulse: true },
    APPROVE: { label: "✅ APPROVE", cls: "bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-emerald-500/30", pulse: false },
  }[decision] || { label: `⬡ ${decision}`, cls: "bg-gray-700 text-gray-300 border-gray-600", pulse: false };

  return (
    <motion.span
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 15 }}
      className={`inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full text-sm font-bold border shadow-lg ${cfg.cls} ${cfg.pulse ? "animate-pulse" : ""}`}
    >
      {cfg.label}
    </motion.span>
  );
}

function MiniScoreBar({ label, value, maxLabel }) {
  const pct = Math.min(100, Math.max(0, value ?? 0));
  const barColor = pct >= 65 ? "bg-red-500" : pct >= 35 ? "bg-amber-500" : "bg-emerald-500";
  
  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-wider w-12">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className={`h-full rounded-full ${barColor}`}
        />
      </div>
      <span className={`text-[9px] font-black tabular-nums w-8 text-right ${
        pct >= 65 ? "text-red-400" : pct >= 35 ? "text-amber-400" : "text-emerald-400"
      }`}>{pct.toFixed(0)}%</span>
    </div>
  );
}

export default function RiskPanel({ tx }) {
  if (!tx) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 py-16 text-gray-600">
        <div className="w-20 h-20 rounded-full border-2 border-dashed border-gray-700 flex items-center justify-center text-3xl">🛡</div>
        <p className="text-sm font-medium">Run a scenario or select a transaction</p>
      </div>
    );
  }

  const riskLevel = tx.risk_score >= 66 ? "HIGH" : tx.risk_score >= 36 ? "MEDIUM" : "LOW";
  const glowClass = tx.risk_score >= 66 ? "shadow-lg shadow-red-900/40" : tx.risk_score >= 36 ? "shadow-lg shadow-amber-900/30" : "shadow-lg shadow-emerald-900/20";

  // Extract score breakdown
  const component_scores = tx.component_scores || {};
  const mlScore = component_scores.ml || 0;
  const ruleScore = Object.keys(component_scores)
    .filter(key => key !== "ml" && !key.startsWith('_') && typeof component_scores[key] === "number")
    .reduce((a, key) => a + component_scores[key], 0) / 5; // average of rule layers

  const finalScore = tx.risk_score || 0;

  return (
    <div className={`flex flex-col items-center gap-4 py-6 px-4 ${glowClass} transition-all duration-500`}>

      {/* Ring gauge */}
      <RiskRing score={tx.risk_score} />

      {/* Risk level text */}
      <div className="text-center">
        <DecisionBadge decision={tx.decision} />
        <p className="text-xs text-gray-500 mt-2 font-semibold tracking-wider uppercase">
          {riskLevel} RISK
        </p>
      </div>

      {/* Score Breakdown mini bars */}
      <div className="w-full px-2 mt-1">
        <MiniScoreBar label="ML" value={mlScore} />
        <MiniScoreBar label="Rules" value={ruleScore} />
        <MiniScoreBar label="Final" value={finalScore} />
      </div>

      {/* Metadata grid */}
      <div className="w-full grid grid-cols-2 gap-2 mt-2">
        {[
          { label: "User", value: tx.user_id },
          { label: "Merchant", value: tx.merchant_id },
          { label: "Amount", value: formatCurrency(tx.amount) },
          { label: "Channel", value: tx.channel?.toUpperCase() || "—" },
          { label: "City", value: tx.city || "—" },
          { label: "TZ", value: tx.device_timezone?.split("/")[1] || "—" },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-800/60 rounded-lg px-3 py-2 border border-gray-700/50">
            <p className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">{label}</p>
            <p className="text-sm text-gray-100 font-semibold truncate mt-0.5">{value || "—"}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
