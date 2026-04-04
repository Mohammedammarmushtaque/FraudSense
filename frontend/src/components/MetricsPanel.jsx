import React from "react";

function MetricCard({ icon, value, label, color, sub }) {
  return (
    <div className={`flex flex-col gap-1 px-4 py-3 rounded-xl bg-gray-800/60 border border-gray-700/50`}>
      <div className="flex items-center gap-2">
        <span className="text-base">{icon}</span>
        <span className={`text-xl font-black tabular-nums ${color}`}>{value}</span>
      </div>
      <p className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">{label}</p>
      {sub && <p className="text-[10px] text-gray-600 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function MetricsPanel({ metrics, transactions }) {
  const txList = transactions || [];
  const blocked = txList.filter((t) => t.decision === "BLOCK").length;
  const held = txList.filter((t) => t.decision === "MFA_HOLD").length;
  const approved = txList.filter((t) => t.decision === "APPROVE").length;
  const total = txList.length;

  const detectionRate = total > 0 ? (((blocked + held) / total) * 100).toFixed(1) : (metrics?.fraud_detection_rate ?? 0);
  const latency = metrics?.avg_latency_ms ?? "—";
  const throughput = metrics?.throughput_tps ?? "—";

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <MetricCard
        icon="⚡"
        value={`${latency}ms`}
        label="Avg Latency"
        color="text-blue-400"
        sub="per transaction"
      />
      <MetricCard
        icon="🔴"
        value={`${detectionRate}%`}
        label="Detection Rate"
        color={parseFloat(detectionRate) > 50 ? "text-red-400" : "text-emerald-400"}
        sub={`${blocked} blocked · ${held} held`}
      />
      <MetricCard
        icon="📊"
        value={total}
        label="Transactions"
        color="text-gray-200"
        sub={`${approved} approved`}
      />
      <MetricCard
        icon="🚀"
        value={typeof throughput === "number" ? throughput.toFixed(2) : throughput}
        label="Throughput"
        color="text-purple-400"
        sub="transactions/sec"
      />
    </div>
  );
}
