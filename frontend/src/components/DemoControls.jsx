import { useState } from "react";
import { Siren, Users, Zap, MessageSquareWarning, TrendingUp } from "lucide-react";

const scenarios = [
  {
    name: "ATO Attack",
    key: "ato_attack",
    icon: Siren,
    color: "bg-red-600 hover:bg-red-700 focus:ring-red-500",
    description: "Account Takeover",
  },
  {
    name: "Coordinated Attack",
    key: "coordinated_attack",
    icon: Users,
    color: "bg-red-700 hover:bg-red-800 focus:ring-red-600",
    description: "Multi-vector Assault",
  },
  {
    name: "Mimicry Attack",
    key: "mimicry_attack",
    icon: Zap,
    color: "bg-red-500 hover:bg-red-600 focus:ring-red-400",
    description: "Behavioral Cloning",
  },
  {
    name: "APP Scam",
    key: "app_scam",
    icon: MessageSquareWarning,
    color: "bg-amber-500 hover:bg-amber-600 focus:ring-amber-400",
    description: "Authorized Push Payment",
  },
  {
    name: "Stealth Probe",
    key: "stealth_probe",
    icon: TrendingUp,
    color: "bg-purple-600 hover:bg-purple-700 focus:ring-purple-500",
    description: "Low & Slow Recon",
  },
];

export default function DemoControls({
  onRunScenario,
  graphIntelligenceEnabled,
  onToggleGraph,
}) {
  const [runningScenario, setRunningScenario] = useState(null);

  const handleRunScenario = async (scenarioKey, scenarioName) => {
    if (runningScenario) return;
    setRunningScenario(scenarioKey);
    try {
      await onRunScenario(scenarioKey);
    } catch (error) {
      console.error(`Failed to run scenario: ${scenarioName}`, error);
    } finally {
      setRunningScenario(null);
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-2xl p-6 shadow-xl w-full max-w-sm">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-white tracking-tight">
          Demo Scenarios
        </h2>
        <p className="text-gray-400 text-sm mt-1">
          Simulate threat patterns to test detection
        </p>
      </div>

      {/* Scenario Buttons */}
      <div className="flex flex-col gap-3 mb-6">
        {scenarios.map(({ name, key, icon: Icon, color, description }) => {
          const isRunning = runningScenario === key;
          const isDisabled = runningScenario !== null;

          return (
            <button
              key={key}
              onClick={() => handleRunScenario(key, name)}
              disabled={isDisabled}
              className={`
                relative flex items-center gap-3 w-full px-4 py-3 rounded-xl
                text-white font-medium text-sm text-left
                transition-all duration-200 ease-in-out
                focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900
                disabled:opacity-60 disabled:cursor-not-allowed
                ${color}
                ${isRunning ? "animate-pulse" : ""}
              `}
              aria-label={`Run ${name} scenario`}
            >
              {/* Icon */}
              <div className="flex-shrink-0">
                {isRunning ? (
                  <svg
                    className="w-5 h-5 animate-spin"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                    />
                  </svg>
                ) : (
                  <Icon className="w-5 h-5" />
                )}
              </div>

              {/* Labels */}
              <div className="flex flex-col leading-tight">
                <span className="font-semibold">
                  {isRunning ? "Running…" : name}
                </span>
                <span className="text-xs opacity-75">{description}</span>
              </div>

              {/* Running indicator dot */}
              {isRunning && (
                <span className="absolute top-2 right-3 flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-white" />
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Divider */}
      <div className="border-t border-gray-700 mb-5" />

      {/* Graph Intelligence Toggle */}
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-white">Graph Intelligence</p>
          <p className="text-xs text-gray-400 mt-0.5">
            {graphIntelligenceEnabled
              ? "Entity relationship analysis active"
              : "Enable for deeper link analysis"}
          </p>
        </div>

        <button
          role="switch"
          aria-checked={graphIntelligenceEnabled}
          onClick={onToggleGraph}
          className={`
            relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full
            border-2 border-transparent transition-colors duration-200 ease-in-out
            focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900 focus:ring-purple-500
            ${graphIntelligenceEnabled ? "bg-purple-600" : "bg-gray-600"}
          `}
        >
          <span
            className={`
              pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow-lg
              transform transition-transform duration-200 ease-in-out
              ${graphIntelligenceEnabled ? "translate-x-5" : "translate-x-0"}
            `}
          />
        </button>
      </div>

      {/* Status Badge */}
      <div
        className={`
          mt-4 flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium
          transition-all duration-300
          ${
            graphIntelligenceEnabled
              ? "bg-purple-900/40 text-purple-300 border border-purple-700/50"
              : "bg-gray-800 text-gray-500 border border-gray-700"
          }
        `}
      >
        <span
          className={`h-1.5 w-1.5 rounded-full flex-shrink-0 ${
            graphIntelligenceEnabled ? "bg-purple-400 animate-pulse" : "bg-gray-600"
          }`}
        />
        {graphIntelligenceEnabled
          ? "Graph engine online — analyzing entity relationships"
          : "Graph engine offline"}
      </div>
    </div>
  );
}