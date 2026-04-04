// frontend/src/utils/ScenarioRunner.js
// Drives demo scenarios and synthetic background traffic against the
// FraudSense API.  All network calls go through the shared apiClient so
// the base-URL and error handling stay in one place.

// ── Constant pools used when generating background traffic ───────────────────

const USERS = ["U001", "U002"];

const MERCHANTS = ["M001", "M002", "M003", "M004", "M005"];

const MERCHANT_CATEGORIES = [
  "groceries",
  "restaurants",
  "electronics",
  "retail",
  "utilities",
];

const CHANNELS = ["web", "mobile"];

const TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT"];

const CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"];

const TIMEZONES = [
  "America/New_York",
  "America/Chicago",
  "America/Denver",
  "America/Los_Angeles",
  "America/Phoenix",
];

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Return a random element from an array.
 * @template T
 * @param {T[]} arr
 * @returns {T}
 */
function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

/**
 * Return a random integer in the range [min, max] (inclusive).
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function randInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Return a random float rounded to 2 decimal places in [min, max].
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
function randFloat(min, max) {
  return parseFloat((Math.random() * (max - min) + min).toFixed(2));
}

/**
 * Generate a collision-resistant transaction ID string.
 * Format: "TX-<timestamp>-<4-digit random>"
 * @returns {string}
 */
function generateTxId() {
  const ts     = Date.now();
  const suffix = String(randInt(1000, 9999));
  return `TX-${ts}-${suffix}`;
}

/**
 * Build a plausible-looking normal transaction payload.
 * Balances are kept internally consistent (newbalanceOrig = old - amount).
 *
 * @returns {Object} Transaction request body matching POST /api/transaction
 */
function buildNormalTransaction() {
  const userId       = pick(USERS);
  const amount       = randFloat(500, 5000);
  const oldBalance   = randFloat(amount + 100, 20000); // always enough funds
  const newBalance   = parseFloat((oldBalance - amount).toFixed(2));
  const merchantId   = pick(MERCHANTS);
  const category     = pick(MERCHANT_CATEGORIES);
  const city         = pick(CITIES);
  const timezone     = pick(TIMEZONES);
  const channel      = pick(CHANNELS);
  const txType       = pick(TX_TYPES);

  return {
    tx_id:              generateTxId(),
    user_id:            userId,
    amount,
    merchant_id:        merchantId,
    merchant_category:  category,
    device_id:          `DEV-${userId}-PRIMARY`,
    city,
    device_timezone:    timezone,
    timestamp:          new Date().toISOString(),
    channel,
    oldbalanceOrg:      oldBalance,
    newbalanceOrig:     newBalance,
    oldbalanceDest:     0,
    newbalanceDest:     amount,
    tx_type:            txType,
  };
}

// ── ScenarioRunner class ─────────────────────────────────────────────────────

export class ScenarioRunner {
  /**
   * @param {Object} [options]
   * @param {Function} [options.onTransactionResult]
   *   Optional callback invoked after every background transaction completes.
   *   Receives (result, payload) so the UI can update the feed in real-time.
   * @param {Function} [options.onBackgroundError]
   *   Optional callback invoked when a background transaction fails.
   *   Receives (error, payload).
   */
  constructor(options = {}) {
    this._onTransactionResult = options.onTransactionResult || null;
    this._onBackgroundError   = options.onBackgroundError   || null;

    // Track active interval IDs so callers can inspect them if needed
    this._activeIntervals = new Set();
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Trigger a named scenario on the backend and return the parsed response.
   *
   * @param {string} scenarioName
   *   One of: 'ato' | 'coordinated' | 'mimicry' | 'scam' | 'stealth_probe'
   * @param {Object} apiClient
   *   Shared API client instance (must expose .runScenario(name)).
   * @returns {Promise<{status: string, scenario: string, transactions: string[]}>}
   * @throws {Error} Re-throws network or API errors so the caller can surface them.
   */
  async runScenario(scenarioName, apiClient) {
    const validScenarios = [
      "ato",
      "coordinated",
      "mimicry",
      "scam",
      "stealth_probe",
    ];

    if (!validScenarios.includes(scenarioName)) {
      throw new Error(
        `Unknown scenario "${scenarioName}". ` +
        `Valid options: ${validScenarios.join(", ")}`
      );
    }

    if (!apiClient || typeof apiClient.runScenario !== "function") {
      throw new Error(
        "apiClient must be an object with a runScenario(name) method."
      );
    }

    try {
      const result = await apiClient.runScenario(scenarioName);
      return result;
    } catch (err) {
      // Enrich the error message before re-throwing
      const enriched = new Error(
        `Scenario "${scenarioName}" failed: ${err.message}`
      );
      enriched.cause = err;
      throw enriched;
    }
  }

  /**
   * Start generating synthetic normal traffic at a fixed interval.
   *
   * Each tick builds a plausible transaction (random user, amount, merchant)
   * and submits it via the internal /transaction endpoint exposed on apiClient.
   *
   * @param {Object}  apiClient
   *   Must expose .submitTransaction(payload) → Promise.
   * @param {number}  [intervalMs=5000]
   *   Milliseconds between successive transactions (minimum 1000 enforced).
   * @returns {number} The interval ID — pass to stopBackgroundTraffic() to halt.
   */
  startBackgroundTraffic(apiClient, intervalMs = 5000) {
    if (!apiClient || typeof apiClient.submitTransaction !== "function") {
      throw new Error(
        "apiClient must expose a submitTransaction(payload) method."
      );
    }

    // Clamp minimum interval to 1 second to avoid hammering the API
    const safeInterval = Math.max(1000, intervalMs);

    const intervalId = setInterval(async () => {
      const payload = buildNormalTransaction();

      try {
        const result = await apiClient.submitTransaction(payload);

        if (typeof this._onTransactionResult === "function") {
          this._onTransactionResult(result, payload);
        }
      } catch (err) {
        // Background traffic errors should never crash the interval —
        // log them and invoke the optional error callback instead.
        console.warn(
          `[ScenarioRunner] Background transaction failed for ` +
          `tx_id=${payload.tx_id}: ${err.message}`
        );

        if (typeof this._onBackgroundError === "function") {
          this._onBackgroundError(err, payload);
        }
      }
    }, safeInterval);

    this._activeIntervals.add(intervalId);
    return intervalId;
  }

  /**
   * Stop a running background-traffic interval.
   *
   * @param {number} intervalId  The value returned by startBackgroundTraffic().
   */
  stopBackgroundTraffic(intervalId) {
    clearInterval(intervalId);
    this._activeIntervals.delete(intervalId);
  }

  /**
   * Convenience method — stop ALL active background-traffic intervals
   * that were started through this ScenarioRunner instance.
   */
  stopAllBackgroundTraffic() {
    for (const id of this._activeIntervals) {
      clearInterval(id);
    }
    this._activeIntervals.clear();
  }

  /**
   * Return the number of currently active background-traffic intervals.
   * Useful for disabling "Start" buttons in the UI.
   *
   * @returns {number}
   */
  get activeIntervalCount() {
    return this._activeIntervals.size;
  }
}

// ── Default singleton export ─────────────────────────────────────────────────
// Components can import the class and instantiate it themselves,
// or import this ready-made singleton for simple use-cases.

export const scenarioRunner = new ScenarioRunner();