/**
 * CaseFilePanel.jsx
 * Displays the AI-generated case file explanation for the selected transaction.
 */

import React from 'react';
import { FileText, Shield, AlertTriangle } from 'lucide-react';

const CaseFilePanel = ({ selectedCase, data, txId }) => {
  if (!data && !selectedCase) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gray-600 gap-3">
        <span className="text-4xl opacity-50">📄</span>
        <p className="text-sm font-medium">Select a flagged transaction...</p>
        <p className="text-[10px] uppercase tracking-wider opacity-40">AI Analysis Pending Selection</p>
      </div>
    );
  }

  const decision = data?.decision || selectedCase?.decision || "N/A";
  const risk = data?.risk_score || 0;

  // AI Summary logic based on current transaction result
  let aiSummary = selectedCase?.summary || "Analyzing latest result...";
  if (risk >= 70) {
    aiSummary = "High-risk transaction detected";
  } else if (risk >= 40) {
    aiSummary = "Medium-risk transaction detected";
  } else if (risk < 40) {
    aiSummary = "Low-risk transaction detected";
  }

  return (
    <div className="flex flex-col gap-5 animate-[fadeIn_0.3s_ease]">
      {/* Summary Header */}
      <div className="flex items-start gap-4 p-4 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
        <div className="p-2.5 rounded-lg bg-indigo-600/20">
          <FileText size={20} className="text-indigo-400" />
        </div>
        <div className="flex-1">
          <p className="text-[10px] font-black text-indigo-400 uppercase tracking-widest mb-1">AI Decision Summary</p>
          <h4 className="text-sm font-bold text-gray-200 leading-snug">{aiSummary}</h4>
        </div>
      </div>

      {/* Risk Factors List */}
      <div className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <AlertTriangle size={14} className="text-amber-500" />
          <h5 className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Contributing Risk Factors</h5>
        </div>
        <ul className="grid grid-cols-1 gap-2">
          {selectedCase?.risk_factors ? (
            selectedCase.risk_factors.map((factor, idx) => (
              <li 
                key={idx}
                className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-gray-800/40 border border-gray-700/40 text-xs text-gray-300"
              >
                <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
                {factor}
              </li>
            ))
          ) : (
            <li className="flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-gray-800/40 border border-gray-700/40 text-xs text-gray-500 italic">
              Detailed risk profiling in progress...
            </li>
          )}
        </ul>
      </div>

      {/* Recommendation Panel */}
      <div className={`mt-2 p-4 rounded-xl border flex items-center justify-between ${
        decision === "BLOCK" 
          ? "bg-red-500/10 border-red-500/20 text-red-400" 
          : decision === "MFA_HOLD"
          ? "bg-amber-500/10 border-amber-500/20 text-amber-400"
          : "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
      }`}>
        <div className="flex items-center gap-3">
          <Shield size={18} />
          <div>
            <p className="text-[10px] font-black uppercase tracking-widest opacity-60">System Action</p>
            <p className="text-xs font-bold">{decision}</p>
          </div>
        </div>
        <span className="text-[10px] font-mono opacity-50">REF: {data?.tx_id || txId}</span>
      </div>
    </div>
  );
};

export default CaseFilePanel;