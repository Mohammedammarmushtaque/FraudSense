import React, { useState, useEffect, useRef, useCallback } from "react";
import { GitBranch, RotateCcw, ChevronRight, Activity, Clock, ArrowRight } from "lucide-react";

// ─── Constants ───────────────────────────────────────────────────────────────

const FSM_STATES = ["CLEAN", "SUSPICIOUS", "CHAIN_ACTIVE", "LOCKED"];

const STATE_CONFIG = {
  CLEAN: {
    color: "bg-green-500",
    border: "border-green-500",
    text: "text-green-700",
    bg: "bg-green-50",
    ring: "ring-green-400",
    badge: "bg-green-100 text-green-800 border-green-300",
    dot: "bg-green-500",
    glow: "shadow-green-200",
  },
  SUSPICIOUS: {
    color: "bg-amber-500",
    border: "border-amber-500",
    text: "text-amber-700",
    bg: "bg-amber-50",
    ring: "ring-amber-400",
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    dot: "bg-amber-500",
    glow: "shadow-amber-200",
  },
  CHAIN_ACTIVE: {
    color: "bg-orange-500",
    border: "border-orange-500",
    text: "text-orange-700",
    bg: "bg-orange-50",
    ring: "ring-orange-400",
    badge: "bg-orange-100 text-orange-800 border-orange-300",
    dot: "bg-orange-500",
    glow: "shadow-orange-200",
  },
  LOCKED: {
    color: "bg-red-600",
    border: "border-red-600",
    text: "text-red-700",
    bg: "bg-red-50",
    ring: "ring-red-500",
    badge: "bg-red-100 text-red-800 border-red-300",
    dot: "bg-red-600",
    glow: "shadow-red-200",
  },
};

const EVENT_TYPE_CONFIG = {
  SUSPICIOUS_TX: { bg: "bg-amber-100", text: "text-amber-700", border: "border-amber-300" },
  CHAIN_TRIGGER: { bg: "bg-orange-100", text: "text-orange-700", border: "border-orange-300" },
  LOCK_TRIGGER: { bg: "bg-red-100", text: "text-red-700", border: "border-red-300" },
  RESET: { bg: "bg-blue-100", text: "text-blue-700", border: "border-blue-300" },
  CLEAN_TX: { bg: "bg-green-100", text: "text-green-700", border: "border-green-300" },
  MFA_SUCCESS: { bg: "bg-teal-100", text: "text-teal-700", border: "border-teal-300" },
  MFA_FAIL: { bg: "bg-rose-100", text: "text-rose-700", border: "border-rose-300" },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

const formatTimestamp = (ts) => {
  if (!ts) return "—";
  const date = new Date(ts);
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  }).format(date);
};

const getEventConfig = (eventType) =>
  EVENT_TYPE_CONFIG[eventType] ?? {
    bg: "bg-slate-100",
    text: "text-slate-600",
    border: "border-slate-300",
  };

// ─── Sub-components ───────────────────────────────────────────────────────────

const FSMBox = ({ label, isActive, isVisited }) => {
  const config = STATE_CONFIG[label] ?? STATE_CONFIG.CLEAN;

  return (
    <div
      className={`
        relative flex flex-col items-center justify-center
        w-28 h-20 rounded-xl border-2 font-semibold text-xs text-center
        transition-all duration-500 ease-in-out select-none
        ${isActive
          ? `${config.border} ${config.bg} ${config.text} ring-4 ${config.ring} ring-offset-2 shadow-lg ${config.glow}`
          : isVisited
          ? `${config.border} ${config.bg} ${config.text} opacity-70`
          : "border-slate-200 bg-slate-50 text-slate-400"
        }
      `}
    >
      {/* Active pulse dot */}
      {isActive && (
        <span className="absolute top-2 right-2">
          <span className={`inline-flex w-2.5 h-2.5 rounded-full ${config.dot} animate-ping opacity-75`} />
          <span className={`absolute inset-0 inline-flex w-2.5 h-2.5 rounded-full ${config.dot}`} />
        </span>
      )}

      {/* Label */}
      <span className={`font-bold text-[11px] tracking-wide ${isActive ? config.text : "text-slate-500"}`}>
        {label.replace("_", " ")}
      </span>

      {isActive && (
        <span className={`mt-1 text-[10px] font-medium px-1.5 py-0.5 rounded-full ${config.badge} border`}>
          CURRENT
        </span>
      )}
    </div>
  );
};

const FSMArrow = ({ fromState, toState, currentState }) => {
  const states = FSM_STATES;
  const fromIdx = states.indexOf(fromState);
  const currentIdx = states.indexOf(currentState);
  const isActive = currentIdx > fromIdx;

  return (
    <div className="flex items-center justify-center mx-1">
      <ChevronRight
        className={`w-5 h-5 transition-colors duration-500 ${
          isActive ? "text-slate-500" : "text-slate-200"
        }`}
      />
    </div>
  );
};

const SuspicionBar = ({ score }) => {
  const clampedScore = Math.max(0, Math.min(100, score ?? 0));

  const getBarColor = () => {
    if (clampedScore < 36) return "bg-green-500";
    if (clampedScore <= 65) return "bg-amber-500";
    return "bg-red-500";
  };

  const getTextColor = () => {
    if (clampedScore < 36) return "text-green-700";
    if (clampedScore <= 65) return "text-amber-700";
    return "text-red-700";
  };

  const getLabel = () => {
    if (clampedScore < 36) return "Low";
    if (clampedScore <= 65) return "Medium";
    return "High";
  };

  return (
    <div className="w-full">
      {/* Score header */}
      <div className="flex items-end justify-between mb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-slate-500" />
          <span className="text-sm font-semibold text-slate-700">Suspicion Score</span>
        </div>
        <div className="flex items-baseline gap-1.5">
          <span className={`text-2xl font-extrabold ${getTextColor()}`}>
            {clampedScore}
          </span>
          <span className="text-xs text-slate-400 font-medium">/100</span>
          <span className={`ml-1 text-xs font-semibold px-2 py-0.5 rounded-full ${
            clampedScore < 36
              ? "bg-green-100 text-green-700"
              : clampedScore <= 65
              ? "bg-amber-100 text-amber-700"
              : "bg-red-100 text-red-700"
          }`}>
            {getLabel()}
          </span>
        </div>
      </div>

      {/* Bar track */}
      <div className="relative h-4 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
        {/* Threshold markers */}
        <div
          className="absolute top-0 bottom-0 w-px bg-green-400 opacity-50 z-10"
          style={{ left: "36%" }}
        />
        <div
          className="absolute top-0 bottom-0 w-px bg-amber-400 opacity-50 z-10"
          style={{ left: "65%" }}
        />

        {/* Fill */}
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${getBarColor()}`}
          style={{ width: `${clampedScore}%` }}
        />
      </div>

      {/* Threshold labels */}
      <div className="relative mt-1 text-[10px] text-slate-400">
        <span className="absolute" style={{ left: "34%" }}>36</span>
        <span className="absolute" style={{ left: "63%" }}>65</span>
      </div>
    </div>
  );
};

const EventLogTable = ({ events }) => {
  const tableRef = useRef(null);

  useEffect(() => {
    if (tableRef.current) {
      tableRef.current.scrollTop = tableRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <div className="w-full">
      <div className="flex items-center gap-2 mb-3">
        <Clock className="w-4 h-4 text-slate-500" />
        <h3 className="text-sm font-semibold text-slate-700">Event Log</h3>
        <span className="ml-auto text-xs text-slate-400 font-medium">
          {events?.length ?? 0} event{events?.length !== 1 ? "s" : ""}
        </span>
      </div>

      <div
        ref={tableRef}
        className="overflow-y-auto rounded-xl border border-slate-200"
        style={{ maxHeight: "260px" }}
      >
        <table className="w-full text-xs border-collapse">
          <thead className="sticky top-0 z-10">
            <tr className="bg-slate-100 border-b border-slate-200">
              <th className="text-left px-3 py-2.5 font-semibold text-slate-600 uppercase tracking-wide text-[11px]">
                #
              </th>
              <th className="text-left px-3 py-2.5 font-semibold text-slate-600 uppercase tracking-wide text-[11px]">
                Event Type
              </th>
              <th className="text-left px-3 py-2.5 font-semibold text-slate-600 uppercase tracking-wide text-[11px]">
                Timestamp
              </th>
              <th className="text-left px-3 py-2.5 font-semibold text-slate-600 uppercase tracking-wide text-[11px]">
                Transition
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 bg-white">
            {!events || events.length === 0 ? (
              <tr>
                <td colSpan={4} className="text-center py-10 text-slate-400 text-xs">
                  No events recorded yet
                </td>
              </tr>
            ) : (
              events.map((event, idx) => {
                const cfg = getEventConfig(event.event_type);
                const fromConfig = STATE_CONFIG[event.from_state] ?? {};
                const toConfig = STATE_CONFIG[event.to_state] ?? {};

                return (
                  <tr
                    key={idx}
                    className="hover:bg-slate-50 transition-colors duration-150"
                  >
                    <td className="px-3 py-2.5 text-slate-400 font-mono">
                      {idx + 1}
                    </td>
                    <td className="px-3 py-2.5">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full font-semibold border ${cfg.bg} ${cfg.text} ${cfg.border}`}
                      >
                        {event.event_type ?? "—"}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 text-slate-500 font-mono whitespace-nowrap">
                      {formatTimestamp(event.timestamp)}
                    </td>
                    <td className="px-3 py-2.5">
                      {event.from_state && event.to_state ? (
                        <div className="flex items-center gap-1.5 flex-wrap">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${fromConfig.badge ?? "bg-slate-100 text-slate-600 border-slate-200"}`}
                          >
                            {event.from_state}
                          </span>
                          <ArrowRight className="w-3 h-3 text-slate-400 shrink-0" />
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${toConfig.badge ?? "bg-slate-100 text-slate-600 border-slate-200"}`}
                          >
                            {event.to_state}
                          </span>
                        </div>
                      ) : (
                        <span className="text-slate-300">—</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────

const ChainMonitor = ({ userId, onResetChain }) => {
  const [chainData, setChainData] = useState({
    state: "CLEAN",
    event_log: [],
    suspicion_score: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [resetting, setResetting] = useState(false);
  const intervalRef = useRef(null);

  const fetchChainData = useCallback(async () => {
    if (!userId) return;
    try {
      const response = await fetch(`/api/chain/${userId}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      const data = await response.json();
      setChainData({
        state: data.state ?? "CLEAN",
        event_log: data.event_log ?? [],
        suspicion_score: data.suspicion_score ?? 0,
      });
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [userId]);

  useEffect(() => {
    setLoading(true);
    fetchChainData();
    intervalRef.current = setInterval(fetchChainData, 2000);
    return () => clearInterval(intervalRef.current);
  }, [fetchChainData]);

  const handleReset = async () => {
    if (!onResetChain || resetting) return;
    setResetting(true);
    try {
      await onResetChain(userId);
      await fetchChainData();
    } finally {
      setTimeout(() => setResetting(false), 800);
    }
  };

  const currentStateIdx = FSM_STATES.indexOf(chainData.state);
  const currentConfig = STATE_CONFIG[chainData.state] ?? STATE_CONFIG.CLEAN;

  return (
    <div className="w-full bg-white rounded-2xl shadow-md border border-slate-200 overflow-hidden">
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-5 py-4 bg-slate-50 border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 bg-slate-200 rounded-lg">
            <GitBranch className="w-4 h-4 text-slate-600" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-slate-800 tracking-wide">
              Chain Monitor
            </h2>
            {userId && (
              <p className="text-[11px] text-slate-400 font-mono">
                User: {userId}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Polling indicator */}
          <div className="flex items-center gap-1.5">
            <span className="relative flex w-2 h-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
            </span>
            <span className="text-[11px] text-slate-400">Live · 2s</span>
          </div>

          {/* Reset button */}
          <button
            onClick={handleReset}
            disabled={resetting || !userId}
            className={`
              flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold
              border transition-all duration-200
              ${resetting || !userId
                ? "border-slate-200 text-slate-400 bg-slate-50 cursor-not-allowed"
                : "border-blue-300 text-blue-700 bg-blue-50 hover:bg-blue-100 hover:border-blue-400 active:scale-95"
              }
            `}
          >
            <RotateCcw className={`w-3.5 h-3.5 ${resetting ? "animate-spin" : ""}`} />
            {resetting ? "Resetting…" : "Reset Chain"}
          </button>
        </div>
      </div>

      {/* ── Error Banner ── */}
      {error && (
        <div className="px-5 py-2.5 bg-red-50 border-b border-red-200 text-xs text-red-600 font-medium flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500 shrink-0" />
          Failed to fetch: {error}
        </div>
      )}

      <div className="p-5 space-y-6">
        {/* ── FSM State Diagram ── */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-700">State Machine</h3>
            {!loading && (
              <div
                className={`
                  inline-flex items-center gap-1.5 px-2.5 py-1
                  rounded-full text-[11px] font-bold border
                  ${currentConfig.badge}
                `}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${currentConfig.dot} animate-pulse`} />
                {chainData.state.replace("_", " ")}
              </div>
            )}
          </div>

          {/* FSM boxes */}
          <div className="flex items-center justify-center flex-wrap gap-1 py-4 px-2 bg-slate-50 rounded-xl border border-slate-100">
            {FSM_STATES.map((state, idx) => (
              <React.Fragment key={state}>
                <FSMBox
                  label={state}
                  isActive={chainData.state === state}
                  isVisited={currentStateIdx > idx}
                />
                {idx < FSM_STATES.length - 1 && (
                  <FSMArrow
                    fromState={state}
                    toState={FSM_STATES[idx + 1]}
                    currentState={chainData.state}
                  />
                )}
              </React.Fragment>
            ))}
          </div>

          {/* State description */}
          <div className={`mt-3 px-3 py-2 rounded-lg text-xs ${currentConfig.bg} ${currentConfig.text} border ${currentConfig.border}`}>
            {chainData.state === "CLEAN" && "✓ No suspicious activity detected. All transactions processing normally."}
            {chainData.state === "SUSPICIOUS" && "⚠ Suspicious patterns detected. Monitoring closely for further anomalies."}
            {chainData.state === "CHAIN_ACTIVE" && "🔗 Chain of suspicious transactions active. Heightened scrutiny applied."}
            {chainData.state === "LOCKED" && "🔒 Account locked due to confirmed fraudulent chain. Manual review required."}
          </div>
        </div>

        {/* ── Suspicion Score Bar ── */}
        <div className="p-4 bg-slate-50 rounded-xl border border-slate-100">
          {loading ? (
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-slate-200 rounded w-1/3" />
              <div className="h-4 bg-slate-200 rounded w-full" />
            </div>
          ) : (
            <SuspicionBar score={chainData.suspicion_score} />
          )}
        </div>

        {/* ── Event Log Table ── */}
        <div>
          {loading ? (
            <div className="animate-pulse space-y-2">
              <div className="h-4 bg-slate-200 rounded w-1/4 mb-3" />
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-9 bg-slate-100 rounded-lg" />
              ))}
            </div>
          ) : (
            <EventLogTable events={chainData.event_log} />
          )}
        </div>

        {/* ── Footer ── */}
        {lastUpdated && (
          <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
            <span>Auto-refreshing every 2 seconds</span>
            <span>Last updated: {lastUpdated.toLocaleTimeString("en-IN")}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChainMonitor;