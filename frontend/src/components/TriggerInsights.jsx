import React from "react";

const FLAG_META = {
  NEW_DEVICE:           { icon: "📱", label: "New Device Login",         color: "border-orange-500/50 text-orange-300 bg-orange-900/20" },
  TIMEZONE_MISMATCH:    { icon: "🌐", label: "VPN / Timezone Mismatch",   color: "border-red-500/50 text-red-300 bg-red-900/20" },
  IMPOSSIBLE_TRAVEL:    { icon: "✈️", label: "Impossible Travel Detected", color: "border-red-500/50 text-red-300 bg-red-900/20" },
  HIGH_VELOCITY:        { icon: "⚡", label: "Suspicious Velocity Pattern",color: "border-amber-500/50 text-amber-300 bg-amber-900/20" },
  CROSS_CHANNEL_BURST:  { icon: "🔀", label: "Cross-Channel Burst",        color: "border-purple-500/50 text-purple-300 bg-purple-900/20" },
  SYNTHETIC_IDENTITY:   { icon: "🤖", label: "Synthetic Identity Suspected",color: "border-red-500/50 text-red-300 bg-red-900/20" },
  FRAUD_LIST_MATCH:     { icon: "🚫", label: "Known Fraud Entity Match",   color: "border-red-500/50 text-red-300 bg-red-900/20" },
  GRADUATED_ESCALATION: { icon: "📈", label: "Graduated Amount Escalation", color: "border-orange-500/50 text-orange-300 bg-orange-900/20" },
  JUST_BELOW_THRESHOLD: { icon: "🔍", label: "Threshold Probing Detected",  color: "border-yellow-500/50 text-yellow-300 bg-yellow-900/20" },
};

const DEFAULT = { icon: "⚠️", color: "border-gray-500/50 text-gray-300 bg-gray-800/40" };

export default function TriggerInsights({ flags = [] }) {
  if (!flags || flags.length === 0) {
    return (
      <div className="flex flex-col gap-3">
        <SectionTitle icon="⚠️" title="Anomaly Triggers" />
        <div className="flex items-center justify-center py-8 text-gray-600 text-sm">
          <span>No anomalies detected — transaction appears clean</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <SectionTitle icon="⚠️" title="Detected Anomalies" count={flags.length} />
      <div className="flex flex-col gap-2">
        {flags.map((flag, i) => {
          const meta = FLAG_META[flag] || { ...DEFAULT, label: flag.replace(/_/g, " ") };
          return (
            <div
              key={i}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl border text-sm font-medium transition-all duration-200 ${meta.color}`}
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <span className="text-lg flex-shrink-0">{meta.icon}</span>
              <span className="flex-1">{meta.label || flag}</span>
              <span className="text-[10px] font-mono opacity-50 flex-shrink-0">{flag}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SectionTitle({ icon, title, count }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-base">{icon}</span>
      <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">{title}</h3>
      {count != null && (
        <span className="ml-auto text-xs font-bold text-red-400 bg-red-900/30 px-2 py-0.5 rounded-full">
          {count} SIGNAL{count !== 1 ? "S" : ""}
        </span>
      )}
    </div>
  );
}
