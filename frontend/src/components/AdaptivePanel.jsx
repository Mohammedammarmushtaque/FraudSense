import React, { useState, useEffect } from "react";

const LAYER_LABELS = {
  device: { label: "Device Layer", icon: "📱" },
  behavioral: { label: "Behavior Layer", icon: "🧠" },
  ml: { label: "ML Engine", icon: "🤖" },
  chain: { label: "Chain Engine", icon: "🔗" },
  graph: { label: "Graph Layer", icon: "🌐" },
};

function WeightRow({ layerKey, newWeight, delta }) {
  const meta = LAYER_LABELS[layerKey] || { label: layerKey, icon: "⬡" };
  const oldWeight = newWeight - (delta || 0);
  const isUp = (delta || 0) > 0;
  const isDown = (delta || 0) < 0;

  return (
    <div className="flex items-center gap-3 py-2 px-3 rounded-lg bg-gray-800/40 border border-gray-700/40">
      <span className="text-base flex-shrink-0">{meta.icon}</span>

      <div className="flex-1 min-w-0">
        <p className="text-xs font-semibold text-gray-300">{meta.label}</p>

        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-[10px] text-gray-500 tabular-nums">
            {(oldWeight * 100).toFixed(0)}%
          </span>

          <span className="text-[10px] text-gray-600">→</span>

          <span className={`text-[10px] font-bold tabular-nums ${isUp ? "text-emerald-400" : isDown ? "text-red-400" : "text-gray-300"
            }`}>
            {(newWeight * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Bar */}
      <div className="relative w-20 h-1.5 bg-gray-700 rounded-full overflow-hidden flex-shrink-0">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{
            width: `${(newWeight * 100)}%`,
            background: isUp ? "#10b981" : isDown ? "#ef4444" : "#6b7280",
          }}
        />
      </div>

      {/* Delta */}
      {delta && delta !== 0 ? (
        <span className={`text-xs font-bold flex-shrink-0 ${isUp ? "text-emerald-400" : "text-red-400"
          }`}>
          {isUp ? "↑" : "↓"} {Math.abs(delta * 100).toFixed(0)}%
        </span>
      ) : (
        <span className="text-xs text-gray-700 flex-shrink-0">—</span>
      )}
    </div>
  );
}

export default function AdaptivePanel({ weightHistory }) {

  const [weights, setWeights] = useState({});
  const [recentChange, setRecentChange] = useState(null);

  // 🔥 Fetch weights from backend
  useEffect(() => {
    const fetchWeights = () => {
      fetch("/api/weights")
        .then(res => res.json())
        .then(data => setWeights(data))
        .catch(err => console.error("Weights fetch error:", err));
    };

    fetchWeights(); // initial
    window.addEventListener("weights-update", fetchWeights);

    const interval = setInterval(fetchWeights, 3000); // every 3s
    return () => {
      clearInterval(interval);
      window.removeEventListener("weights-update", fetchWeights);
    };
  }, []);

  // Existing change highlight logic
  useEffect(() => {
    if (weightHistory && weightHistory.length > 0) {
      const last = weightHistory[weightHistory.length - 1];
      setRecentChange(last);

      const timer = setTimeout(() => setRecentChange(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [weightHistory]);

  return (
    <div className="flex flex-col gap-4">

      {/* Header */}
      <div className="flex items-center gap-2">
        <span className="text-base">🔁</span>
        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">
          Adaptive Learning
        </h3>

        {recentChange && (
          <span className="ml-auto text-xs font-bold text-emerald-400 bg-emerald-900/30 border border-emerald-700/50 px-2 py-0.5 rounded-full animate-pulse">
            WEIGHTS UPDATED
          </span>
        )}
      </div>

      {/* Alert */}
      {recentChange && (
        <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-emerald-900/20 border border-emerald-700/40">
          <span className="text-xl">🧠</span>
          <div>
            <p className="text-xs font-bold text-emerald-300">System Learning Update</p>
            <p className="text-[11px] text-emerald-500/80 mt-0.5">
              Analyst feedback processed — detection weights recalibrated
            </p>
          </div>
        </div>
      )}

      {/* 🔥 LIVE WEIGHTS */}
      {weights && Object.keys(weights).length > 0 ? (
        <div className="flex flex-col gap-2">
          {Object.entries(weights).map(([layer, weight]) => {
            const hist = weightHistory?.find((h) => h.layer === layer);

            return (
              <WeightRow
                key={layer}
                layerKey={layer}
                newWeight={(weight || 0) / 100}
                delta={hist?.delta || 0}
              />
            );
          })}
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {[
            { k: "device", w: 0.20 },
            { k: "behavioral", w: 0.30 },
            { k: "ml", w: 0.20 },
            { k: "chain", w: 0.15 },
            { k: "graph", w: 0.15 },
          ].map(({ k, w }) => (
            <WeightRow key={k} layerKey={k} newWeight={w} delta={0} />
          ))}
        </div>
      )}

      <p className="text-[10px] text-gray-700 text-center mt-1">
        Weights adjust automatically via analyst fraud confirmation feedback
      </p>
    </div>
  );
}