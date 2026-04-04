// frontend/src/components/RiskGauge.jsx
// Displays risk score as a radial gauge, decision badge, and chain-state
// FSM mini-visual.  All layout is Tailwind; chart is Recharts.

import React, { useMemo } from "react";
import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
  ResponsiveContainer,
} from "recharts";

// ── Constants ────────────────────────────────────────────────────────────────

/** Ordered FSM states used to draw the mini-pipeline visual. */
const FSM_STATES = [
  "CLEAR",
  "DEVICE_FLAGGED",
  "TRANSACTION_ATTEMPTED",
  "MFA_CHALLENGED",
  "BLOCKED",
  "CONFIRMED_FRAUD",
];

/**
 * Map each FSM state to a short display label that fits inside the small node.
 */
const STATE_LABELS = {
  CLEAR:                  "CLEAR",
  DEVICE_FLAGGED:         "DEV",
  TRANSACTION_ATTEMPTED:  "TXN",
  MFA_CHALLENGED:         "MFA",
  BLOCKED:                "BLK",
  CONFIRMED_FRAUD:        "FRAUD",
};

/**
 * Decide which risk "tier" a score falls into.
 * @param {number} score
 * @returns {"low"|"medium"|"high"}
 */
function riskTier(score) {
  if (score < 36) return "low";
  if (score <= 65) return "medium";
  return "high";
}

// ── Sub-components ───────────────────────────────────────────────────────────

/**
 * The coloured fill arc drawn inside the RadialBarChart.
 * Pulled out so we can unit-test colour logic independently.
 */
function GaugeFill({ score }) {
  const tier = riskTier(score);

  const fillColor = {
    low:    "#22c55e",   // Tailwind green-500
    medium: "#f59e0b",   // Tailwind amber-500
    high:   "#ef4444",   // Tailwind red-500
  }[tier];

  // RadialBarChart expects data as an array; value drives the arc length.
  const data = [{ value: score, fill: fillColor }];

  return (
    <RadialBarChart
      width={220}
      height={220}
      innerRadius={70}
      outerRadius={100}
      startAngle={225}
      endAngle={-45}
      data={data}
      barSize={16}
    >
      {/* Full-circle grey track so the gauge always looks "full" */}
      <PolarAngleAxis
        type="number"
        domain={[0, 100]}
        angleAxisId={0}
        tick={false}
      />
      <RadialBar
        background={{ fill: "#1f2937" }}   // Tailwind gray-800 track
        dataKey="value"
        angleAxisId={0}
        cornerRadius={8}
      />
    </RadialBarChart>
  );
}

/**
 * Large centred score number overlaid on top of the gauge SVG.
 */
function ScoreOverlay({ score }) {
  const tier = riskTier(score);

  const textColor = {
    low:    "text-green-400",
    medium: "text-amber-400",
    high:   "text-red-400",
  }[tier];

  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
      <span className={`text-5xl font-black tabular-nums ${textColor}`}>
        {Math.round(score)}
      </span>
      <span className="text-xs text-gray-400 mt-1 uppercase tracking-widest">
        Risk Score
      </span>
    </div>
  );
}

/**
 * Coloured pill badge showing the decision label.
 */
function DecisionBadge({ decision }) {
  const styles = {
    APPROVE:  {
      wrapper: "bg-green-900/40 border border-green-500/50 text-green-300",
      dot:     "bg-green-400",
      icon:    "✓",
    },
    MFA_HOLD: {
      wrapper: "bg-amber-900/40 border border-amber-500/50 text-amber-300",
      dot:     "bg-amber-400",
      icon:    "⚠",
    },
    BLOCK:    {
      wrapper: "bg-red-900/40 border border-red-500/50 text-red-300",
      dot:     "bg-red-400",
      icon:    "✕",
    },
  };

  const s = styles[decision] ?? {
    wrapper: "bg-gray-800 border border-gray-600 text-gray-400",
    dot:     "bg-gray-500",
    icon:    "?",
  };

  return (
    <div
      className={`
        flex items-center gap-2 px-5 py-2 rounded-full text-sm font-bold
        uppercase tracking-widest shadow-lg ${s.wrapper}
      `}
    >
      <span
        className={`w-2.5 h-2.5 rounded-full animate-pulse ${s.dot}`}
        aria-hidden="true"
      />
      <span className="text-lg">{s.icon}</span>
      <span>{decision ?? "UNKNOWN"}</span>
    </div>
  );
}

/**
 * Mini FSM pipeline: a row of state nodes connected by arrows, with the
 * active state highlighted.
 */
function ChainStateVisual({ chainState }) {
  // Normalise incoming state to uppercase for matching
  const active = (chainState ?? "").toUpperCase();

  return (
    <div className="w-full">
      {/* Label */}
      <p className="text-xs text-gray-500 uppercase tracking-widest text-center mb-3">
        Chain State
      </p>

      {/* Node pipeline */}
      <div className="flex items-center justify-center gap-0.5 flex-wrap">
        {FSM_STATES.map((state, idx) => {
          const isActive = active === state;
          const isPast   = FSM_STATES.indexOf(active) > idx;

          return (
            <React.Fragment key={state}>
              {/* State node */}
              <div
                title={state}
                className={`
                  relative flex items-center justify-center
                  w-10 h-10 rounded-lg text-[10px] font-bold
                  transition-all duration-300 cursor-default select-none
                  ${isActive
                    ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/40 scale-110 z-10"
                    : isPast
                      ? "bg-gray-700 text-gray-400"
                      : "bg-gray-800 text-gray-600 border border-gray-700"
                  }
                `}
              >
                {/* Pulse ring on active node */}
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-lg border-2 border-indigo-400 animate-ping opacity-60"
                    aria-hidden="true"
                  />
                )}
                {STATE_LABELS[state]}
              </div>

              {/* Arrow connector — skip after last node */}
              {idx < FSM_STATES.length - 1 && (
                <div
                  className={`
                    w-3 h-px flex-shrink-0
                    ${isPast || isActive ? "bg-indigo-500" : "bg-gray-700"}
                  `}
                  aria-hidden="true"
                />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Human-readable active state label */}
      <p className="text-center text-xs text-indigo-400 mt-3 font-mono tracking-wide">
        {active || "UNKNOWN"}
      </p>
    </div>
  );
}

// ── Main export ──────────────────────────────────────────────────────────────

/**
 * RiskGauge
 *
 * @param {object}  props
 * @param {number}  props.riskScore   0–100 risk score
 * @param {string}  props.decision    "APPROVE" | "MFA_HOLD" | "BLOCK"
 * @param {string}  props.chainState  One of the FSM_STATES strings
 */
export default function RiskGauge({
  riskScore  = 0,
  decision   = "APPROVE",
  chainState = "CLEAR",
}) {
  // Clamp score to [0, 100] so malformed data never breaks the chart
  const safeScore = useMemo(
    () => Math.min(100, Math.max(0, Number(riskScore) || 0)),
    [riskScore]
  );

  const tier = riskTier(safeScore);

  // Subtle glow class applied to the card border
  const glowClass = {
    low:    "shadow-green-900/20",
    medium: "shadow-amber-900/30",
    high:   "shadow-red-900/40 border-red-900/40",
  }[tier];

  return (
    <div
      className={`
        flex flex-col items-center gap-6 p-6
        bg-gray-900 border border-gray-800 rounded-2xl
        shadow-xl ${glowClass}
        transition-all duration-500
      `}
    >
      {/* ── Gauge ── */}
      <div className="relative w-[220px] h-[220px]">
        <GaugeFill score={safeScore} />
        <ScoreOverlay score={safeScore} />
      </div>

      {/* ── Decision badge ── */}
      <DecisionBadge decision={decision} />

      {/* ── Divider ── */}
      <div className="w-full border-t border-gray-800" />

      {/* ── Chain state FSM visual ── */}
      <ChainStateVisual chainState={chainState} />
    </div>
  );
}