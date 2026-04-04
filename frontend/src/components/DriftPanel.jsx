import React, { useState, useEffect } from "react";
import { TrendingUp, AlertOctagon, RefreshCw, Shield } from "lucide-react";

const ALERT_CONFIG = {
  BOUNDARY_PROBING: {
    color: "amber",
    bgClass: "bg-amber-50 border-amber-200",
    badgeClass: "bg-amber-100 text-amber-800 border border-amber-300",
    iconClass: "text-amber-500",
    headerClass: "bg-amber-100",
    icon: TrendingUp,
    label: "Boundary Probing",
  },
  EVASION_IMPROVING: {
    color: "red",
    bgClass: "bg-red-50 border-red-200",
    badgeClass: "bg-red-100 text-red-800 border border-red-300",
    iconClass: "text-red-500",
    headerClass: "bg-red-100",
    icon: AlertOctagon,
    label: "Evasion Improving",
  },
};

const DEFAULT_CONFIG = {
  bgClass: "bg-gray-50 border-gray-200",
  badgeClass: "bg-gray-100 text-gray-800 border border-gray-300",
  iconClass: "text-gray-500",
  headerClass: "bg-gray-100",
  icon: AlertOctagon,
  label: "Unknown",
};

function formatTimestamp(timestamp) {
  try {
    const date = new Date(timestamp);
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date);
  } catch {
    return timestamp;
  }
}

function AlertCard({ alert, index }) {
  const config = ALERT_CONFIG[alert.type] || DEFAULT_CONFIG;
  const IconComponent = config.icon;

  return (
    <div
      className={`rounded-xl border-2 ${config.bgClass} overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-200`}
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Card Header */}
      <div className={`${config.headerClass} px-4 py-3 flex items-center justify-between`}>
        <div className="flex items-center gap-2">
          <IconComponent className={`w-5 h-5 ${config.iconClass}`} strokeWidth={2.5} />
          <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${config.badgeClass}`}>
            {config.label}
          </span>
        </div>
        <span className="text-xs text-gray-500 font-mono">
          {formatTimestamp(alert.timestamp)}
        </span>
      </div>

      {/* Card Body */}
      <div className="px-4 py-4 space-y-3">
        {/* Message */}
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
            Alert
          </p>
          <p className="text-sm text-gray-800 leading-relaxed">{alert.message}</p>
        </div>

        {/* Recommendation */}
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
            Recommendation
          </p>
          <p className="text-sm text-gray-700 leading-relaxed italic">
            {alert.recommendation}
          </p>
        </div>
      </div>
    </div>
  );
}

function RetrainButton({ onClick, isRetraining }) {
  return (
    <button
      onClick={onClick}
      disabled={isRetraining}
      className="flex items-center gap-2 px-5 py-2.5 bg-red-600 hover:bg-red-700 disabled:bg-red-400
                 text-white text-sm font-semibold rounded-lg shadow-md hover:shadow-lg
                 transition-all duration-200 disabled:cursor-not-allowed"
    >
      <RefreshCw
        className={`w-4 h-4 ${isRetraining ? "animate-spin" : ""}`}
        strokeWidth={2.5}
      />
      {isRetraining ? "Retraining..." : "Initiate Retraining"}
    </button>
  );
}

export default function DriftPanel() {
  const [driftAlerts, setDriftAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isRetraining, setIsRetraining] = useState(false);
  const [retrainStatus, setRetrainStatus] = useState(null);

  const fetchAlerts = async () => {
    try {
      const response = await fetch("/api/drift/alerts");
      if (!response.ok) {
        throw new Error(`HTTP error: ${response.status}`);
      }
      const data = await response.json();
      setDriftAlerts(Array.isArray(data) ? data : data.alerts || []);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err.message || "Failed to fetch alerts");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 5000);
    return () => clearInterval(interval);
  }, []);

  const hasEvasionAlert = driftAlerts.some(
    (alert) => alert.type === "EVASION_IMPROVING"
  );

  const handleRetrain = async () => {
    setIsRetraining(true);
    setRetrainStatus(null);
    try {
      const response = await fetch("/api/model/retrain", { method: "POST" });
      if (!response.ok) throw new Error("Retrain request failed");
      setRetrainStatus({ type: "success", message: "Retraining initiated successfully." });
    } catch (err) {
      setRetrainStatus({ type: "error", message: err.message || "Retrain failed." });
    } finally {
      setIsRetraining(false);
    }
  };

  const boundaryCount = driftAlerts.filter((a) => a.type === "BOUNDARY_PROBING").length;
  const evasionCount = driftAlerts.filter((a) => a.type === "EVASION_IMPROVING").length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-5xl mx-auto">

        {/* Panel Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-indigo-600 rounded-xl shadow-lg">
                <Shield className="w-7 h-7 text-white" strokeWidth={2} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white tracking-tight">
                  Threat Intelligence
                </h1>
                <p className="text-slate-400 text-sm mt-0.5">
                  Real-time adversarial drift monitoring
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 flex-wrap">
              {/* Retrain Button */}
              {hasEvasionAlert && (
                <RetrainButton onClick={handleRetrain} isRetraining={isRetraining} />
              )}

              {/* Live Indicator */}
              <div className="flex items-center gap-2 bg-slate-700 px-3 py-2 rounded-lg">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-green-500" />
                </span>
                <span className="text-xs text-slate-300 font-medium">Live · 5s</span>
              </div>
            </div>
          </div>

          {/* Retrain Status */}
          {retrainStatus && (
            <div
              className={`mt-4 px-4 py-3 rounded-lg text-sm font-medium ${
                retrainStatus.type === "success"
                  ? "bg-green-900/40 text-green-300 border border-green-700"
                  : "bg-red-900/40 text-red-300 border border-red-700"
              }`}
            >
              {retrainStatus.message}
            </div>
          )}
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="bg-slate-700/50 rounded-xl p-4 border border-slate-600/50">
            <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1">
              Total Alerts
            </p>
            <p className="text-3xl font-bold text-white">{driftAlerts.length}</p>
          </div>
          <div className="bg-amber-900/30 rounded-xl p-4 border border-amber-700/40">
            <p className="text-amber-400 text-xs font-semibold uppercase tracking-wider mb-1">
              Boundary Probing
            </p>
            <p className="text-3xl font-bold text-amber-300">{boundaryCount}</p>
          </div>
          <div className="bg-red-900/30 rounded-xl p-4 border border-red-700/40">
            <p className="text-red-400 text-xs font-semibold uppercase tracking-wider mb-1">
              Evasion Improving
            </p>
            <p className="text-3xl font-bold text-red-300">{evasionCount}</p>
          </div>
        </div>

        {/* Content Area */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 text-slate-400">
            <RefreshCw className="w-10 h-10 animate-spin mb-4 text-indigo-400" />
            <p className="text-lg font-medium">Fetching threat intelligence...</p>
          </div>
        )}

        {error && !loading && (
          <div className="flex flex-col items-center justify-center py-16 bg-red-900/20 border border-red-700/40 rounded-xl">
            <AlertOctagon className="w-10 h-10 text-red-400 mb-3" />
            <p className="text-red-300 font-semibold text-lg">Connection Error</p>
            <p className="text-red-400 text-sm mt-1">{error}</p>
            <button
              onClick={fetchAlerts}
              className="mt-4 px-4 py-2 bg-red-700 hover:bg-red-600 text-white text-sm rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && driftAlerts.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 bg-slate-700/30 border border-slate-600/30 rounded-xl">
            <Shield className="w-12 h-12 text-green-400 mb-4" />
            <p className="text-slate-200 font-semibold text-xl">All Clear</p>
            <p className="text-slate-400 text-sm mt-2">No active drift alerts detected.</p>
          </div>
        )}

        {!loading && !error && driftAlerts.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {driftAlerts.map((alert, index) => (
              <AlertCard key={`${alert.type}-${alert.timestamp}-${index}`} alert={alert} index={index} />
            ))}
          </div>
        )}

        {/* Footer */}
        {lastUpdated && (
          <div className="mt-6 text-center text-slate-500 text-xs">
            Last updated: {formatTimestamp(lastUpdated.toISOString())}
          </div>
        )}
      </div>
    </div>
  );
}