// frontend/src/components/TopBar.jsx
// Top navigation bar displaying system title and key fraud detection metrics

import React from 'react';
import { Shield, Activity, AlertTriangle, Brain } from 'lucide-react';

/**
 * Individual metric card displayed in the center of the TopBar
 */
const MetricCard = ({ icon: Icon, label, value, iconColor, valueColor }) => (
  <div className="flex items-center gap-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl px-4 py-2.5 shadow-inner">
    <div className={`p-1.5 rounded-lg bg-white/10 ${iconColor}`}>
      <Icon size={16} />
    </div>
    <div className="flex flex-col">
      <span className="text-white/60 text-xs font-medium uppercase tracking-wide leading-none mb-1">
        {label}
      </span>
      <span className={`text-sm font-bold leading-none ${valueColor}`}>
        {value}
      </span>
    </div>
  </div>
);

/**
 * TopBar component - Main navigation header for the FraudSense dashboard
 *
 * @param {Object} props.metrics - System metrics object
 * @param {number} props.metrics.total_tx - Total number of transactions processed
 * @param {number} props.metrics.fraud_rate - Current fraud rate as a decimal (e.g. 0.043)
 * @param {number} props.metrics.avg_risk - Average risk score across transactions (0-1)
 * @param {string} props.metrics.model_version - Current ML model version string
 */
const TopBar = ({ metrics }) => {
  // Safely destructure metrics with fallback defaults
  const {
    total_tx = 0,
    fraud_rate = 0,
    avg_risk = 0,
    model_version = 'N/A',
  } = metrics || {};

  // Format total transactions with locale-aware comma separators
  const formattedTotalTx = total_tx.toLocaleString();

  // Format fraud rate as percentage with 2 decimal places
  const formattedFraudRate = `${(fraud_rate * 100).toFixed(2)}%`;

  // Format average risk score as percentage with 1 decimal place
  const formattedAvgRisk = `${(avg_risk * 100).toFixed(1)}%`;

  // Determine fraud rate color based on severity thresholds
  const fraudRateColor =
    fraud_rate > 0.1
      ? 'text-red-300'
      : fraud_rate > 0.05
        ? 'text-yellow-300'
        : 'text-emerald-300';

  // Determine avg risk color based on severity thresholds
  const avgRiskColor =
    avg_risk > 0.7
      ? 'text-red-300'
      : avg_risk > 0.4
        ? 'text-yellow-300'
        : 'text-emerald-300';

  return (
    <header className="w-full bg-gradient-to-r from-slate-900 via-blue-950 to-slate-900 shadow-2xl border-b border-white/10">
      <div className="max-w-screen-2xl mx-auto px-6 py-3">
        <div className="flex items-center justify-between gap-6">

          {/* ── Left: Brand Title ── */}
          <div className="flex items-center gap-3 flex-shrink-0">
            {/* Shield icon with glow effect */}
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500 rounded-xl blur-md opacity-40" />
              <div className="relative bg-gradient-to-br from-blue-500 to-blue-700 p-2.5 rounded-xl shadow-lg">
                <Shield size={22} className="text-white" strokeWidth={2.5} />
              </div>
            </div>

            {/* Brand name and subtitle */}
            <div className="flex flex-col">
              <span className="text-white text-xl font-extrabold tracking-tight leading-none">
                FraudSense
              </span>
              <span className="text-blue-300/70 text-xs font-medium tracking-widest uppercase leading-none mt-1">
                Detection System
              </span>
            </div>
          </div>

          {/* ── Center: Metric Cards ── */}
          <div className="flex items-center gap-3 flex-1 justify-center">
            {/* Total Transactions */}
            <MetricCard
              icon={Activity}
              label="Total Transactions"
              value={formattedTotalTx}
              iconColor="text-blue-300"
              valueColor="text-white"
            />

            {/* Divider */}
            <div className="w-px h-8 bg-white/20 flex-shrink-0" />

            {/* Fraud Rate */}
            <MetricCard
              icon={AlertTriangle}
              label="Fraud Rate"
              value={formattedFraudRate}
              iconColor={fraudRateColor}
              valueColor={fraudRateColor}
            />

            {/* Divider */}
            <div className="w-px h-8 bg-white/20 flex-shrink-0" />

            {/* Average Risk Score */}
            <MetricCard
              icon={Brain}
              label="Avg Risk Score"
              value={formattedAvgRisk}
              iconColor={avgRiskColor}
              valueColor={avgRiskColor}
            />
          </div>

          {/* ── Right: Model Version Badge ── */}
          <div className="flex items-center flex-shrink-0">
            <div className="flex items-center gap-2 bg-gradient-to-r from-violet-600/30 to-blue-600/30 border border-violet-400/30 rounded-full px-4 py-2 shadow-lg backdrop-blur-sm">
              {/* Pulsing status indicator */}
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-violet-300" />
              </span>
              <span className="text-violet-200 text-xs font-semibold tracking-wide">
                Model v{model_version}
              </span>
            </div>
          </div>

        </div>
      </div>
    </header>
  );
};

export default TopBar;