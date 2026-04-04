/**
 * AIDecisionPanel.jsx
 * Shows ML decision reasoning with score breakdown (ML + Rule + Final).
 * Color-coded by decision: SAFE=Green, MFA=Yellow, BLOCK=Red.
 */

import React from "react";
import { motion } from "framer-motion";

function ScoreBar({ label, score, color, icon }) {
  const pct = Math.min(100, Math.max(0, score));
  return (
    <div className="flex items-center gap-2">
      <span className="text-[9px] font-bold text-gray-500 uppercase tracking-widest w-16 flex items-center gap-1">
        {icon} {label}
      </span>
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={`h-full rounded-full ${color}`}
        />
      </div>
      <span className={`text-[10px] font-black tabular-nums w-12 text-right ${
        pct > 65 ? "text-red-400" : pct > 35 ? "text-amber-400" : "text-emerald-400"
      }`}>{pct.toFixed(1)}%</span>
    </div>
  );
}

export default function AIDecisionPanel({ tx }) {
  if (!tx) {
    return (
      <div className="flex flex-col items-center justify-center p-6 text-gray-500">
        <span className="text-2xl mb-2 flex items-center justify-center">🧠</span>
        <span className="text-xs font-semibold">Awaiting transaction context...</span>
      </div>
    );
  }

  const score = tx.risk_score ?? 0;
  const displayScore = score <= 1 ? Math.round(score * 100) : Math.round(score);
  const csVal = tx.component_scores ?? {};
  const mlProbability = tx.fraud_probability ?? (csVal._fraud_probability ?? 0);
  const mlExplanations = tx.ml_explanations ?? (csVal._ml_explanations ?? []);

  // Score breakdown from decision engine
  const component_scores = tx.component_scores || {};
  const mlScore = component_scores.ml || 0;
  const ruleScore = Object.keys(component_scores)
    .filter(key => key !== "ml" && !key.startsWith('_') && typeof component_scores[key] === "number")
    .reduce((a, key) => a + component_scores[key], 0) / 5; // average of rule layers

  const finalScore = tx.risk_score || 0;

  // Parse component scores
  const parseScore = (k1, k2) => {
    let v = csVal[k1] ?? csVal[k2] ?? 0;
    if (v <= 1 && v > 0) v = v * 100;
    return v;
  };

  const amtScore = parseScore("amount_score", "amount");
  const locScore = parseScore("location_score", "location");
  const devScore = parseScore("device_score", "device");
  const behScore = parseScore("behavioral_score", "behavioral");

  const factors = [];

  if (amtScore >= 70) {
    factors.push({
      label: "Critical Amount Deviation",
      desc: "Transaction value exceptionally higher than user baseline",
      icon: "💸", level: "high"
    });
  } else if (amtScore >= 40) {
    factors.push({
      label: "Elevated Amount",
      desc: "Transaction size moderately exceeds historical run-rate",
      icon: "💰", level: "medium"
    });
  }

  if (locScore >= 60) {
    factors.push({
      label: "Suspicious Location",
      desc: `Origin location (${tx.city || "Unknown"}) flags anomaly`,
      icon: "📍", level: "high"
    });
  }

  if (devScore >= 65) {
    factors.push({
      label: "Unrecognized Device",
      desc: "Device fingerprint has no prior relationship with user",
      icon: "📱", level: "high"
    });
  }

  if (behScore >= 60) {
    factors.push({
      label: "Behavioral Mismatch",
      desc: "Velocity or category falls outside predicted sequence",
      icon: "👤", level: "high"
    });
  }

  if (mlExplanations.length > 0) {
    mlExplanations.slice(0, 2).forEach(exp => {
      factors.push({
        label: exp,
        desc: `ML RandomForest confidence: ${(mlProbability <= 1 ? mlProbability * 100 : mlProbability).toFixed(1)}%`,
        icon: "🤖",
        level: mlProbability > 0.7 ? "high" : mlProbability > 0.4 ? "medium" : "low"
      });
    });
  }

  // Use reasons from backend
  if (factors.length === 0 && tx.reasons?.length > 0) {
    tx.reasons.slice(0, 3).forEach(f => {
      factors.push({
        label: f,
        desc: "Risk pipeline trigger detected",
        icon: "⚠️",
        level: "medium"
      });
    });
  }
  
  if (factors.length === 0 && tx.flags?.length > 0) {
    tx.flags.slice(0, 3).forEach(f => {
      factors.push({
        label: f,
        desc: "Rule-based trigger detected",
        icon: "⚠️",
        level: "medium"
      });
    });
  }

  if (factors.length === 0) {
    factors.push({
      label: "Profile Match",
      desc: "Transaction perfectly aligns with historical baselines",
      icon: "✅", level: "low"
    });
  }

  const levelW = { high: 3, medium: 2, low: 1 };
  factors.sort((a, b) => levelW[b.level] - levelW[a.level]);

  const riskCategory = displayScore >= 66 ? "HIGH RISK" : displayScore >= 36 ? "MEDIUM RISK" : "LOW RISK";
  const catTheme =
    riskCategory === "HIGH RISK"
      ? "text-red-400 bg-red-500/10 border-red-500/30"
      : riskCategory === "MEDIUM RISK"
      ? "text-amber-400 bg-amber-500/10 border-amber-500/30"
      : "text-emerald-400 bg-emerald-500/10 border-emerald-500/30";

  // Decision color theme
  const decisionTheme = tx.decision === "BLOCK"
    ? "border-red-500/40 bg-red-500/5"
    : tx.decision === "MFA_HOLD"
    ? "border-amber-500/40 bg-amber-500/5"
    : "border-emerald-500/40 bg-emerald-500/5";

  return (
    <div className="flex flex-col gap-4">
      {/* ── Summary Stats ── */}
      <div className="flex gap-3">
        <div className={`flex-1 flex justify-between items-center px-4 py-3 rounded-xl border ${catTheme} shadow-inner`}>
          <div className="flex flex-col">
            <span className="text-[9px] font-bold uppercase tracking-widest opacity-80 mb-0.5">Risk Category</span>
            <span className="text-sm font-black tracking-wide">{riskCategory}</span>
          </div>
          <span className="text-2xl opacity-80">
            {riskCategory.includes("HIGH") ? "🛑" : riskCategory.includes("MEDIUM") ? "⚠️" : "✅"}
          </span>
        </div>

        <div className="flex-1 flex justify-between items-center px-4 py-3 rounded-xl border border-indigo-500/30 bg-indigo-500/10 text-indigo-300 shadow-inner">
          <div className="flex flex-col">
            <span className="text-[9px] font-bold uppercase tracking-widest opacity-80 mb-0.5">ML Confidence</span>
            <span className="text-sm font-black tracking-wide">{(mlProbability <= 1 ? mlProbability * 100 : mlProbability).toFixed(1)}%</span>
          </div>
          <span className="text-2xl opacity-80">🤖</span>
        </div>
      </div>

      {/* ── Score Breakdown (ML + Rule + Final) ── */}
      <div className={`p-3 rounded-xl border ${decisionTheme}`}>
        <div className="flex items-center gap-2 mb-2.5">
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Score Breakdown</span>
          <div className="h-px bg-gray-800 flex-1"></div>
          <span className={`text-[10px] font-black px-2 py-0.5 rounded-full border ${
            tx.decision === "BLOCK" ? "bg-red-900/40 text-red-400 border-red-700/50"
            : tx.decision === "MFA_HOLD" ? "bg-amber-900/40 text-amber-400 border-amber-700/50"
            : "bg-emerald-900/40 text-emerald-400 border-emerald-700/50"
          }`}>{tx.decision}</span>
        </div>
        <div className="flex flex-col gap-2">
          <ScoreBar label="ML" score={mlScore} color="bg-indigo-500" icon="🧠" />
          <ScoreBar label="Rules" score={ruleScore} color="bg-cyan-500" icon="📋" />
          <div className="border-t border-gray-800/60 pt-2 mt-0.5">
            <ScoreBar label="Final" score={finalScore} color={
              finalScore > 70 ? "bg-red-500" : finalScore > 40 ? "bg-amber-500" : "bg-emerald-500"
            } icon="⚡" />
          </div>
        </div>
        <p className="text-[9px] text-gray-600 mt-2 text-center">
          Final = Rules + ML
        </p>
      </div>

      {/* ── Decision Reasoning ── */}
      <div className="flex flex-col gap-2.5">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">
            Decision Reasoning
          </span>
          <div className="h-px bg-gray-800 flex-1"></div>
        </div>
        
        <div className="flex flex-col gap-2">
          {factors.slice(0, 5).map((f, i) => {
            const iconBg =
              f.level === "high" ? "bg-red-500/20 text-red-400 border-red-500/30"
                : f.level === "medium" ? "bg-amber-500/20 text-amber-400 border-amber-500/30"
                : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";

            return (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.1, duration: 0.3 }}
                className="flex items-start gap-3 p-2.5 rounded-lg bg-gray-900/40 border border-gray-800 hover:border-gray-700 transition-colors"
              >
                <div className={`mt-0.5 w-7 h-7 rounded border flex shrink-0 items-center justify-center text-sm ${iconBg}`}>
                  {f.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-bold text-gray-200 truncate">{f.label}</p>
                  <p className="text-[10px] text-gray-400 leading-snug mt-0.5">{f.desc}</p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
