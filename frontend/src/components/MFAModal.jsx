import { useState, useEffect, useRef, useCallback } from "react";
import { AlertTriangle, Clock } from "lucide-react";

const COUNTDOWN_SECONDS = 30;

export default function MFAModal({ isOpen, txData, onVerify, onCancel }) {
  const [otpCode, setOtpCode] = useState(["", "", "", ""]);
  const [purpose, setPurpose] = useState("");
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);
  const [isVerifying, setIsVerifying] = useState(false);
  const [shake, setShake] = useState(false);

  const inputRefs = useRef([]);
  const intervalRef = useRef(null);
  const hasAutoCancelled = useRef(false);

  // ── Reset state whenever modal opens ──────────────────────────────────────
  useEffect(() => {
    if (isOpen) {
      setOtpCode(["", "", "", ""]);
      setPurpose("");
      setCountdown(COUNTDOWN_SECONDS);
      setIsVerifying(false);
      hasAutoCancelled.current = false;

      // Auto-focus first OTP box after transition
      const t = setTimeout(() => inputRefs.current[0]?.focus(), 120);
      return () => clearTimeout(t);
    }
  }, [isOpen]);

  // ── Countdown ticker ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!isOpen) {
      clearInterval(intervalRef.current);
      return;
    }

    intervalRef.current = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(intervalRef.current);
          if (!hasAutoCancelled.current) {
            hasAutoCancelled.current = true;
            onCancel();
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(intervalRef.current);
  }, [isOpen, onCancel]);

  // ── OTP helpers ───────────────────────────────────────────────────────────
  const handleOtpChange = useCallback((index, value) => {
    const digit = value.replace(/\D/g, "").slice(-1);
    setOtpCode((prev) => {
      const next = [...prev];
      next[index] = digit;
      return next;
    });
    if (digit && index < 3) {
      inputRefs.current[index + 1]?.focus();
    }
  }, []);

  const handleOtpKeyDown = useCallback((index, e) => {
    if (e.key === "Backspace") {
      if (otpCode[index] === "" && index > 0) {
        inputRefs.current[index - 1]?.focus();
        setOtpCode((prev) => {
          const next = [...prev];
          next[index - 1] = "";
          return next;
        });
      } else {
        setOtpCode((prev) => {
          const next = [...prev];
          next[index] = "";
          return next;
        });
      }
    }
    if (e.key === "ArrowLeft" && index > 0) inputRefs.current[index - 1]?.focus();
    if (e.key === "ArrowRight" && index < 3) inputRefs.current[index + 1]?.focus();
  }, [otpCode]);

  const handleOtpPaste = useCallback((e) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 4);
    if (!pasted) return;
    const next = ["", "", "", ""];
    [...pasted].forEach((ch, i) => { next[i] = ch; });
    setOtpCode(next);
    const focusIdx = Math.min(pasted.length, 3);
    inputRefs.current[focusIdx]?.focus();
  }, []);

  // ── Verify ────────────────────────────────────────────────────────────────
  const handleVerify = useCallback(async () => {
    const code = otpCode.join("");
    if (code.length < 4) {
      setShake(true);
      setTimeout(() => setShake(false), 600);
      return;
    }
    if (txData?.requires_purpose && !purpose.trim()) {
      setShake(true);
      setTimeout(() => setShake(false), 600);
      return;
    }

    setIsVerifying(true);
    clearInterval(intervalRef.current);
    try {
      await onVerify(code, purpose.trim());
    } catch {
      setIsVerifying(false);
    }
  }, [otpCode, purpose, txData, onVerify]);

  // ── Formatting helpers ────────────────────────────────────────────────────
  const formatAmount = (amount) =>
    typeof amount === "number"
      ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(amount)
      : amount ?? "—";

  const countdownColor =
    countdown < 10 ? "text-red-400" : countdown < 20 ? "text-amber-400" : "text-gray-400";

  const countdownRingFraction = countdown / COUNTDOWN_SECONDS;
  const radius = 10;
  const circumference = 2 * Math.PI * radius;
  const strokeDash = circumference * countdownRingFraction;

  if (!isOpen) return null;

  return (
    // ── Backdrop ────────────────────────────────────────────────────────────
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="mfa-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
    >
      {/* Blur overlay */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onCancel}
        aria-hidden="true"
      />

      {/* Modal panel */}
      <div
        className={`
          relative z-10 w-full max-w-md
          bg-gray-900 border-2 border-amber-500/70
          rounded-2xl shadow-2xl shadow-amber-900/30
          animate-[modalIn_0.2s_ease-out]
          ${shake ? "animate-[shake_0.5s_ease-in-out]" : ""}
        `}
        style={{
          animation: shake
            ? "shake 0.5s ease-in-out"
            : "modalIn 0.2s ease-out",
        }}
      >
        {/* Amber glow top bar */}
        <div className="h-1 w-full bg-gradient-to-r from-amber-600 via-amber-400 to-amber-600 rounded-t-xl" />

        <div className="p-6">
          {/* ── Header ──────────────────────────────────────────────────── */}
          <div className="flex items-start justify-between mb-5">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-amber-500/15 border border-amber-500/40 flex-shrink-0">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
              </div>
              <div>
                <h2
                  id="mfa-title"
                  className="text-white font-bold text-lg leading-tight"
                >
                  Verify Transaction
                </h2>
                <p className="text-gray-400 text-xs mt-0.5">
                  Additional authentication required
                </p>
              </div>
            </div>

            {/* Countdown ring */}
            <div className="flex flex-col items-center gap-1 flex-shrink-0">
              <div className="relative w-9 h-9">
                <svg className="w-9 h-9 -rotate-90" viewBox="0 0 24 24">
                  <circle
                    cx="12" cy="12" r={radius}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="text-gray-700"
                  />
                  <circle
                    cx="12" cy="12" r={radius}
                    fill="none"
                    strokeWidth="2"
                    stroke={countdown < 10 ? "#f87171" : countdown < 20 ? "#fbbf24" : "#6b7280"}
                    strokeDasharray={`${strokeDash} ${circumference}`}
                    strokeLinecap="round"
                    style={{ transition: "stroke-dasharray 0.9s linear, stroke 0.3s" }}
                  />
                </svg>
                <span className={`absolute inset-0 flex items-center justify-center text-[10px] font-bold ${countdownColor}`}>
                  {countdown}
                </span>
              </div>
              <span className={`text-[10px] font-medium ${countdownColor} flex items-center gap-0.5`}>
                <Clock className="w-2.5 h-2.5" />
                sec
              </span>
            </div>
          </div>

          {/* ── Transaction Details ─────────────────────────────────────── */}
          <div className="bg-gray-800/60 border border-gray-700/60 rounded-xl p-4 mb-5 space-y-2.5">
            <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-1">
              Transaction Details
            </p>

            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Transaction ID</span>
              <span className="text-gray-200 text-sm font-mono">
                {txData?.tx_id ?? "—"}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Amount</span>
              <span className="text-amber-400 text-sm font-bold">
                {formatAmount(txData?.amount)}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-gray-400 text-sm">Merchant</span>
              <span className="text-gray-200 text-sm font-mono truncate max-w-[160px]">
                {txData?.merchant_id ?? "—"}
              </span>
            </div>
          </div>

          {/* ── Purpose Input (conditional) ─────────────────────────────── */}
          {txData?.requires_purpose && (
            <div className="mb-5">
              <label
                htmlFor="purpose-input"
                className="block text-sm font-medium text-gray-300 mb-2"
              >
                Why are you making this transfer?
                <span className="text-red-400 ml-1">*</span>
              </label>
              <textarea
                id="purpose-input"
                value={purpose}
                onChange={(e) => setPurpose(e.target.value)}
                placeholder="e.g. Paying my landlord for rent…"
                rows={2}
                maxLength={200}
                disabled={isVerifying}
                className="
                  w-full px-3 py-2.5 rounded-lg text-sm
                  bg-gray-800 border border-gray-600
                  text-white placeholder-gray-500
                  focus:outline-none focus:ring-2 focus:ring-amber-500/60 focus:border-amber-500
                  disabled:opacity-50 disabled:cursor-not-allowed
                  resize-none transition-colors duration-150
                "
              />
              <p className="text-right text-xs text-gray-600 mt-1">
                {purpose.length}/200
              </p>
            </div>
          )}

          {/* ── OTP Input ───────────────────────────────────────────────── */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              One-Time Passcode
              <span className="text-red-400 ml-1">*</span>
            </label>

            <div className="flex gap-3 justify-center" onPaste={handleOtpPaste}>
              {otpCode.map((digit, index) => (
                <input
                  key={index}
                  ref={(el) => (inputRefs.current[index] = el)}
                  type="text"
                  inputMode="numeric"
                  pattern="[0-9]*"
                  maxLength={1}
                  value={digit}
                  onChange={(e) => handleOtpChange(index, e.target.value)}
                  onKeyDown={(e) => handleOtpKeyDown(index, e)}
                  disabled={isVerifying}
                  aria-label={`OTP digit ${index + 1}`}
                  className="
                    w-14 h-14 text-center text-2xl font-bold
                    bg-gray-800 border-2 rounded-xl
                    text-white caret-amber-400
                    transition-all duration-150
                    focus:outline-none focus:border-amber-500 focus:ring-2 focus:ring-amber-500/30 focus:bg-gray-750
                    disabled:opacity-50 disabled:cursor-not-allowed
                    placeholder-gray-600
                    border-gray-600
                    [appearance:textfield]
                  "
                  style={{
                    borderColor: digit ? "#f59e0b" : undefined,
                    boxShadow: digit ? "0 0 0 1px rgba(245,158,11,0.2)" : undefined,
                  }}
                />
              ))}
            </div>

            <p className="text-center text-xs text-gray-500 mt-3">
              Enter the 4-digit code sent to your registered device
            </p>
          </div>

          {/* ── Action Buttons ──────────────────────────────────────────── */}
          <div className="flex gap-3">
            <button
              onClick={onCancel}
              disabled={isVerifying}
              className="
                flex-1 px-4 py-3 rounded-xl text-sm font-semibold
                bg-gray-800 text-gray-300 border border-gray-700
                hover:bg-gray-700 hover:text-white hover:border-gray-600
                focus:outline-none focus:ring-2 focus:ring-gray-500/50
                disabled:opacity-40 disabled:cursor-not-allowed
                transition-all duration-150
              "
            >
              Cancel
            </button>

            <button
              onClick={handleVerify}
              disabled={isVerifying || otpCode.join("").length < 4 || countdown === 0}
              className="
                flex-[2] px-4 py-3 rounded-xl text-sm font-semibold
                bg-amber-500 text-gray-900
                hover:bg-amber-400
                focus:outline-none focus:ring-2 focus:ring-amber-500/60 focus:ring-offset-2 focus:ring-offset-gray-900
                disabled:opacity-40 disabled:cursor-not-allowed
                transition-all duration-150
                flex items-center justify-center gap-2
              "
            >
              {isVerifying ? (
                <>
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  Verifying…
                </>
              ) : (
                "Verify & Authorize"
              )}
            </button>
          </div>

          {/* ── Footer note ─────────────────────────────────────────────── */}
          <p className="text-center text-xs text-gray-600 mt-4">
            This session expires in{" "}
            <span className={`font-semibold ${countdownColor}`}>
              {countdown}s
            </span>
            . Never share your OTP with anyone.
          </p>
        </div>
      </div>

      {/* Keyframe styles injected via style tag */}
      <style>{`
        @keyframes modalIn {
          from { opacity: 0; transform: scale(0.95) translateY(-8px); }
          to   { opacity: 1; transform: scale(1)    translateY(0);     }
        }
        @keyframes shake {
          0%, 100% { transform: translateX(0);   }
          15%       { transform: translateX(-6px); }
          30%       { transform: translateX(6px);  }
          45%       { transform: translateX(-4px); }
          60%       { transform: translateX(4px);  }
          75%       { transform: translateX(-2px); }
          90%       { transform: translateX(2px);  }
        }
      `}</style>
    </div>
  );
}