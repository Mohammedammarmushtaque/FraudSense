import React from "react";

const LAYERS = [
  { key: "amount_score",    label: "Amount Risk",   icon: "💰", color: "#f59e0b" },
  { key: "location_score",  label: "Location Risk", icon: "📍", color: "#ef4444" },
  { key: "device_score",    label: "Device Risk",   icon: "📱", color: "#f97316" },
  { key: "behavioral_score",label: "Behavior Risk", icon: "🧠", color: "#8b5cf6" },
  { key: "ml_score",        label: "ML Risk",       icon: "🤖", color: "#3b82f6" },
  { key: "chain_score",     label: "Chain Risk",    icon: "🔗", color: "#10b981" },
];

function ScoreBar({ value, color, label, icon }) {
  const pct = Math.min(100, Math.max(0, (value || 0)));
  const displayPct = pct.toFixed(0);
  const riskLabel = pct >= 66 ? "HIGH" : pct >= 36 ? "MED" : "LOW";
  const riskColor = pct >= 66 ? "text-red-400" : pct >= 36 ? "text-amber-400" : "text-emerald-400";

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm">{icon}</span>
          <span className="text-xs font-semibold text-gray-400">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-bold uppercase ${riskColor}`}>{riskLabel}</span>
          <span className="text-xs font-black text-white tabular-nums w-8 text-right">{displayPct}</span>
        </div>
      </div>
      <div className="relative h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            boxShadow: pct > 50 ? `0 0 8px ${color}88` : "none",
          }}
        />
      </div>
    </div>
  );
}

export default function PipelineBreakdown({ pipelineScores, finalScore }) {
  if (!pipelineScores) {
    return (
      <div className="flex flex-col gap-3">
        <SectionTitle />
        <div className="flex flex-col gap-3 mt-2">
          {LAYERS.map((layer) => (
            <ScoreBar key={layer.key} value={0} color={layer.color} label={layer.label} icon={layer.icon} />
          ))}
        </div>
        <p className="text-xs text-gray-600 text-center mt-1">Run a scenario to populate pipeline scores</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <SectionTitle />
      <div className="flex flex-col gap-3">
        {LAYERS.map((layer) => (
          <ScoreBar
            key={layer.key}
            value={pipelineScores[layer.key] || 0}
            color={layer.color}
            label={layer.label}
            icon={layer.icon}
          />
        ))}
      </div>

      {finalScore != null && (
        <div className="mt-2 pt-3 border-t border-gray-800 flex items-center justify-between">
          <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Composite Score</span>
          <div className={`text-xl font-black tabular-nums ${
            finalScore >= 66 ? "text-red-400" : finalScore >= 36 ? "text-amber-400" : "text-emerald-400"
          }`}>
            {finalScore.toFixed(1)}
            <span className="text-xs text-gray-500 font-normal ml-1">/100</span>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="flex items-center gap-3 text-[10px] text-gray-600 pt-1">
        {[{ l: "LOW", c: "text-emerald-600" }, { l: "MED", c: "text-amber-600" }, { l: "HIGH", c: "text-red-600" }].map(({ l, c }) => (
          <span key={l} className={`font-bold ${c}`}>{l}</span>
        ))}
        <span className="flex-1" />
        <span className="text-gray-700">Score 0–100 scale per layer</span>
      </div>
    </div>
  );
}

function SectionTitle() {
  return (
    <div className="flex items-center gap-2">
      <span className="text-base">📊</span>
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Pipeline Breakdown</h3>
    </div>
  );
}
