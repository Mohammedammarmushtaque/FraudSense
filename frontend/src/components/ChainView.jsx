import React from "react";

const STATE_CONFIG = {
  CLEAN:        { color: "text-emerald-400", border: "border-emerald-500/40", bg: "bg-emerald-900/20", dot: "bg-emerald-400" },
  WATCH:        { color: "text-amber-400",   border: "border-amber-500/40",   bg: "bg-amber-900/20",   dot: "bg-amber-400 animate-pulse" },
  MFA_REQUIRED: { color: "text-orange-400",  border: "border-orange-500/40",  bg: "bg-orange-900/20",  dot: "bg-orange-400 animate-pulse" },
  BLOCKED:      { color: "text-red-400",     border: "border-red-500/40",     bg: "bg-red-900/20",     dot: "bg-red-400 animate-pulse" },
};

const CHAIN_STEPS = [
  { event: "LOGIN_NEW_DEVICE",      label: "New Device Login",         icon: "📱" },
  { event: "TRANSACTION_ATTEMPT",   label: "Suspicious Transaction",   icon: "💸" },
  { event: "FAILED_MFA",            label: "MFA Challenge Failed",     icon: "🔐" },
  { event: "ANALYST_CONFIRM_FRAUD", label: "Fraud Confirmed",          icon: "🚨" },
  { event: "MFA_SUCCESS",           label: "MFA Verified",             icon: "✅" },
];

function parsedEvents(eventLog) {
  if (!eventLog) return [];
  try {
    return typeof eventLog === "string" ? JSON.parse(eventLog) : eventLog;
  } catch {
    return [];
  }
}

export default function ChainView({ chainState, eventLog }) {
  const cfg = STATE_CONFIG[chainState] || STATE_CONFIG.CLEAN;
  const events = parsedEvents(eventLog);

  // Build sequence from log — keep only known events in order
  const knownEventKeys = new Set(CHAIN_STEPS.map((s) => s.event));
  const seenEvents = events.filter((e) => knownEventKeys.has(e.event));

  const sequenceDisplay = seenEvents.length > 0
    ? seenEvents.map((e) => {
        const step = CHAIN_STEPS.find((s) => s.event === e.event);
        return step ? `${step.icon} ${step.label}` : e.event;
      })
    : null;

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <span className="text-base">🔗</span>
        <h3 className="text-xs font-bold text-gray-400 uppercase tracking-widest">Fraud Chain Sequence</h3>
        <span className={`ml-auto flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold border ${cfg.border} ${cfg.bg} ${cfg.color}`}>
          <span className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${cfg.dot}`} />
          {chainState || "CLEAN"}
        </span>
      </div>

      {/* Chain progression */}
      {sequenceDisplay && sequenceDisplay.length > 0 ? (
        <div className="relative flex flex-col gap-0">
          {sequenceDisplay.map((step, i) => {
            const isLast = i === sequenceDisplay.length - 1;
            return (
              <div key={i} className="flex items-start gap-3">
                {/* Timeline line */}
                <div className="flex flex-col items-center flex-shrink-0 w-6">
                  <div className={`w-3 h-3 rounded-full border-2 flex-shrink-0 mt-3 ${
                    isLast
                      ? chainState === "BLOCKED" ? "bg-red-500 border-red-400" : "bg-amber-500 border-amber-400"
                      : "bg-gray-600 border-gray-500"
                  }`} />
                  {!isLast && <div className="w-px flex-1 bg-gray-700 mt-1" style={{ minHeight: "24px" }} />}
                </div>
                {/* Step label */}
                <div className={`flex-1 text-sm py-2.5 px-3 rounded-lg mt-1 mb-0.5 border transition-all ${
                  isLast
                    ? chainState === "BLOCKED"
                      ? "bg-red-900/30 border-red-700/40 text-red-300 font-semibold"
                      : "bg-amber-900/30 border-amber-700/40 text-amber-300 font-semibold"
                    : "bg-gray-800/50 border-gray-700/30 text-gray-400"
                }`}>
                  {step}
                  {isLast && chainState === "BLOCKED" && (
                    <span className="ml-2 text-xs font-bold text-red-400 uppercase tracking-wider">← CURRENT</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center py-8 gap-3 text-gray-600">
          {chainState === "WATCH" ? (
            <>
              <div className="w-12 h-12 rounded-full bg-amber-900/20 border border-amber-700/40 flex items-center justify-center text-2xl animate-pulse">👁</div>
              <p className="text-sm text-amber-500 font-semibold">Account under surveillance</p>
              <p className="text-xs text-gray-500">Monitoring for next suspicious action</p>
            </>
          ) : (
            <>
              <div className="w-12 h-12 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center text-2xl">✓</div>
              <p className="text-sm text-emerald-500 font-semibold">No fraud chain detected</p>
            </>
          )}
        </div>
      )}

      {/* Event count */}
      {events.length > 0 && (
        <div className="text-xs text-gray-600 text-right font-mono">
          {events.length} chain event{events.length !== 1 ? "s" : ""} recorded
        </div>
      )}
    </div>
  );
}
