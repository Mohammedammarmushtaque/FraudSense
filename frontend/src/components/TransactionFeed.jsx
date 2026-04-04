import React, { useState, useEffect } from "react";
import { CheckCircle, AlertCircle, XCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const getRiskScoreStyle = (score) => {
  if (score < 36) return { bg: "bg-green-100", text: "text-green-800", border: "border-green-300" };
  if (score <= 65) return { bg: "bg-amber-100", text: "text-amber-800", border: "border-amber-300" };
  return { bg: "bg-red-100", text: "text-red-800", border: "border-red-300" };
};

const getDecisionConfig = (decision) => {
  switch (decision) {
    case "APPROVE":
      return {
        bg: "bg-green-500",
        text: "text-white",
        icon: <CheckCircle className="w-3.5 h-3.5" />,
        label: "APPROVE",
      };
    case "MFA_HOLD":
      return {
        bg: "bg-amber-500",
        text: "text-white",
        icon: <AlertCircle className="w-3.5 h-3.5" />,
        label: "MFA HOLD",
      };
    case "BLOCK":
      return {
        bg: "bg-red-500",
        text: "text-white",
        icon: <XCircle className="w-3.5 h-3.5" />,
        label: "BLOCK",
      };
    default:
      return {
        bg: "bg-gray-400",
        text: "text-white",
        icon: <AlertCircle className="w-3.5 h-3.5" />,
        label: decision,
      };
  }
};

const getChainStateConfig = (state) => {
  switch (state) {
    case "SUSPICIOUS":
      return { bg: "bg-orange-100", text: "text-orange-700", border: "border-orange-300" };
    case "FLAGGED":
      return { bg: "bg-red-100", text: "text-red-700", border: "border-red-300" };
    case "MONITORED":
      return { bg: "bg-blue-100", text: "text-blue-700", border: "border-blue-300" };
    default:
      return { bg: "bg-gray-100", text: "text-gray-600", border: "border-gray-300" };
  }
};

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

const formatAmount = (amount) => {
  if (amount === undefined || amount === null) return "₹—";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(amount);
};

const FlagPill = ({ flag }) => (
  <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200 transition-colors duration-150 hover:bg-slate-200">
    {flag}
  </span>
);

const recordFeedback = async (txId, isFraud) => {
  try {
    await fetch("http://localhost:8000/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tx_id: txId, is_fraud: isFraud }),
    });
    window.dispatchEvent(new Event("weights-update"));
  } catch (err) {
    console.error("Feedback error:", err);
  }
};

const TransactionRow = ({ tx }) => {
  const riskStyle = getRiskScoreStyle(tx.risk_score ?? 0);
  const decisionConfig = getDecisionConfig(tx.decision);
  const showChainBadge = tx.chain_state && tx.chain_state !== "CLEAN";
  const chainConfig = showChainBadge ? getChainStateConfig(tx.chain_state) : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, backgroundColor: "#f8fafc" }}
      animate={{ opacity: 1, y: 0, backgroundColor: "#ffffff" }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.4 }}
      className="
        group flex items-start gap-4 px-5 py-4
        border-b border-slate-100 last:border-b-0
        hover:bg-slate-50
        transition-all duration-200 ease-in-out
        cursor-default
      "
    >
      {/* LEFT — tx_id & timestamp */}
      <div className="flex flex-col gap-1 min-w-[170px] max-w-[190px]">
        <span
          className="text-xs font-mono font-semibold text-slate-800 truncate"
          title={tx.tx_id}
        >
          {tx.tx_id ?? "—"}
        </span>
        <span className="text-[11px] text-slate-400 leading-tight">
          {formatTimestamp(tx.created_at || tx.timestamp)}
        </span>
      </div>

      {/* CENTER — user_id, amount, merchant_id */}
      <div className="flex flex-col gap-1 flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 font-medium">
            User:
          </span>
          <span className="text-xs font-semibold text-slate-700 truncate">
            {tx.user_id ?? "—"}
          </span>
        </div>

        <span className="text-base font-bold text-slate-900 tracking-tight">
          {formatAmount(tx.amount)}
        </span>

        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 font-medium">
            Merchant:
          </span>
          <span className="text-xs font-semibold text-slate-600 truncate">
            {tx.merchant_id ?? "—"}
          </span>
        </div>

        {/* Feedback Buttons */}
        <div className="flex gap-2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button 
            onClick={() => recordFeedback(tx.tx_id, 1)}
            className="text-[10px] font-bold text-red-600 bg-red-50 px-2 py-1 rounded border border-red-200 hover:bg-red-100"
          >
            MARK FRAUD
          </button>
          <button 
            onClick={() => recordFeedback(tx.tx_id, 0)}
            className="text-[10px] font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded border border-emerald-200 hover:bg-emerald-100"
          >
            MARK SAFE
          </button>
        </div>

        {/* Flags */}
        {tx.flags && tx.flags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-1">
            {tx.flags.map((flag, idx) => (
              <FlagPill key={idx} flag={flag} />
            ))}
          </div>
        )}
      </div>

      {/* RIGHT — risk score, decision, chain state */}
      <div className="flex flex-col items-end gap-2 shrink-0">
        {/* Risk Score Badge */}
        <div
          className={`
            inline-flex items-center gap-1.5 px-2.5 py-1
            rounded-full border text-xs font-bold
            ${riskStyle.bg} ${riskStyle.text} ${riskStyle.border}
            transition-transform duration-150 group-hover:scale-105
          `}
        >
          <span className="text-[10px] font-medium opacity-70">Risk</span>
          <span>{tx.risk_score ?? "—"}</span>
        </div>

        {/* Decision Badge */}
        <div
          className={`
            inline-flex items-center gap-1.5 px-2.5 py-1
            rounded-full text-xs font-bold
            ${decisionConfig.bg} ${decisionConfig.text}
            shadow-sm transition-transform duration-150 group-hover:scale-105
          `}
        >
          {decisionConfig.icon}
          <span>{decisionConfig.label}</span>
        </div>

        {/* Chain State Badge (only if not CLEAN) */}
        {showChainBadge && (
          <div
            className={`
              inline-flex items-center gap-1 px-2 py-0.5
              rounded border text-[11px] font-semibold
              ${chainConfig.bg} ${chainConfig.text} ${chainConfig.border}
              transition-transform duration-150 group-hover:scale-105
            `}
          >
            <span>⛓</span>
            <span>{tx.chain_state}</span>
          </div>
        )}
      </div>
    </motion.div>
  );
};

const TransactionFeed = () => {
  const [transactions, setTransactions] = useState([]);

  const fetchTransactions = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/transactions");
      const data = await res.json();
      setTransactions(prev => {
        const prevMap = new Map(prev.map(t => [t.tx_id, t]));
        for (const dbTx of data) {
          if (!prevMap.has(dbTx.tx_id)) {
            prevMap.set(dbTx.tx_id, dbTx);
          }
        }
        return Array.from(prevMap.values())
          .sort((a,b) => new Date(b.created_at || b.timestamp).getTime() - new Date(a.created_at || a.timestamp).getTime())
          .slice(0, 50);
      });
    } catch (err) {
      console.error("Fetch transactions error:", err);
    }
  };

  useEffect(() => {
    fetchTransactions();
    const interval = setInterval(fetchTransactions, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full bg-white rounded-2xl shadow-md border border-slate-200 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200 bg-slate-50">
        <div className="flex items-center gap-2">
          <div className="w-2.5 h-2.5 rounded-full bg-green-400 animate-pulse" />
          <h2 className="text-sm font-semibold text-slate-700 tracking-wide uppercase">
            Transaction Feed
          </h2>
        </div>
        <span className="text-xs text-slate-400 font-medium">
          {transactions.length} transaction{transactions.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 px-5 py-2.5 bg-slate-50 border-b border-slate-100 flex-wrap">
        <span className="text-[11px] text-slate-400 font-semibold uppercase tracking-wide">
          Risk:
        </span>
        {[
          { label: "Low (<36)", bg: "bg-green-100", text: "text-green-700" },
          { label: "Medium (36–65)", bg: "bg-amber-100", text: "text-amber-700" },
          { label: "High (>65)", bg: "bg-red-100", text: "text-red-700" },
        ].map((item) => (
          <span
            key={item.label}
            className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${item.bg} ${item.text}`}
          >
            {item.label}
          </span>
        ))}
      </div>

      {/* Scrollable List */}
      <div
        className="overflow-y-auto divide-y divide-slate-100 scroll-smooth"
        style={{ maxHeight: "600px" }}
      >
        {transactions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 text-slate-400">
            <XCircle className="w-10 h-10 mb-3 opacity-40" />
            <p className="text-sm font-medium">No transactions to display</p>
            <p className="text-xs mt-1 opacity-70">Transactions will appear here in real time</p>
          </div>
        ) : (
          <AnimatePresence initial={false}>
            {transactions.map((tx, idx) => (
              <TransactionRow key={tx.tx_id ?? idx} tx={tx} />
            ))}
          </AnimatePresence>
        )}
      </div>

      {/* Footer */}
      {transactions.length > 0 && (
        <div className="px-5 py-2.5 bg-slate-50 border-t border-slate-200 text-right">
          <span className="text-[11px] text-slate-400">
            Showing {transactions.length} record{transactions.length !== 1 ? "s" : ""}
          </span>
        </div>
      )}
    </div>
  );
};

export default TransactionFeed;