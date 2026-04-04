// frontend/src/api/client.js
// Centralized API client — all calls go through Vite proxy → backend:8000

import axios from 'axios';

// Use relative path so Vite proxy (vite.config.js) routes /api → localhost:8000
const API_BASE_URL = '/api';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`[API Request] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API Request Error]', error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging
apiClient.interceptors.response.use(
  (response) => {
    console.log(`[API Response] ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('[API Response Error]', error?.response?.status, error?.response?.data);
    return Promise.reject(error);
  }
);

/**
 * Fetch a list of transactions
 * @param {number} limit - Maximum number of transactions to retrieve (default: 50)
 * @returns {Promise<Object>} Transaction data from the API
 */
export const fetchTransactions = async (limit = 50) => {
  try {
    const response = await apiClient.get('/transactions', {
      params: { limit },
    });
    return response.data;
  } catch (error) {
    console.error('[fetchTransactions] Failed to fetch transactions:', error?.message);
    throw error;
  }
};

/**
 * Fetch system metrics and statistics
 * @returns {Promise<Object>} Metrics data from the API
 */
export const fetchMetrics = async () => {
  try {
    const response = await apiClient.get('/metrics');
    return response.data;
  } catch (error) {
    console.error('[fetchMetrics] Failed to fetch metrics:', error?.message);
    throw error;
  }
};

/**
 * Fetch a detailed case file for a specific transaction
 * @param {string} txId - The transaction ID to retrieve the case file for
 * @returns {Promise<Object>} Case file data from the API
 */
export const fetchCaseFile = async (txId) => {
  try {
    const response = await apiClient.get(`/case/${txId}`);
    return response.data;
  } catch (error) {
    console.error(`[fetchCaseFile] Failed to fetch case file for transaction ${txId}:`, error?.message);
    throw error;
  }
};

/**
 * Fetch the blockchain state for a specific user
 * @param {string} userId - The user ID to retrieve chain state for
 * @returns {Promise<Object>} Chain state data from the API
 */
export const fetchChainState = async (userId) => {
  try {
    const response = await apiClient.get(`/chain/${userId}`);
    return response.data;
  } catch (error) {
    console.error(`[fetchChainState] Failed to fetch chain state for user ${userId}:`, error?.message);
    throw error;
  }
};

/**
 * Trigger a coordinated fraud pattern check
 * @returns {Promise<Object>} Coordinated check results from the API
 */
export const fetchCoordinatedCheck = async () => {
  try {
    const response = await apiClient.post('/coordinated/check');
    return response.data;
  } catch (error) {
    console.error('[fetchCoordinatedCheck] Failed to perform coordinated check:', error?.message);
    throw error;
  }
};

/**
 * Fetch active drift alerts from the system
 * @returns {Promise<Object>} Drift alerts data from the API
 */
export const fetchDriftAlerts = async () => {
  try {
    const response = await apiClient.get('/drift/alerts');
    return response.data;
  } catch (error) {
    console.error('[fetchDriftAlerts] Failed to fetch drift alerts:', error?.message);
    throw error;
  }
};

/**
 * Verify multi-factor authentication for a transaction
 * @param {string} txId - The transaction ID requiring MFA verification
 * @param {string} otpCode - The one-time password code provided by the user
 * @param {string} purpose - The purpose or context of the MFA verification
 * @returns {Promise<Object>} MFA verification result from the API
 */
export const verifyMFA = async (txId, otpCode, purpose) => {
  try {
    const response = await apiClient.post('/mfa/verify', {
      tx_id: txId,
      otp_code: otpCode,
      purpose,
    });
    return response.data;
  } catch (error) {
    console.error(`[verifyMFA] Failed to verify MFA for transaction ${txId}:`, error?.message);
    throw error;
  }
};

/**
 * Submit an analyst's fraud confirmation decision
 * @param {string} txId - The transaction ID being reviewed
 * @param {boolean} isFraud - Whether the analyst confirms the transaction as fraudulent
 * @param {string} analystId - The ID of the analyst submitting the confirmation
 * @returns {Promise<Object>} Confirmation result from the API
 */
export const confirmFraud = async (txId, isFraud, analystId) => {
  try {
    const response = await apiClient.post('/analyst/confirm', {
      tx_id: txId,
      is_fraud: isFraud,
      analyst_id: analystId,
    });
    return response.data;
  } catch (error) {
    console.error(`[confirmFraud] Failed to confirm fraud status for transaction ${txId}:`, error?.message);
    throw error;
  }
};

/**
 * Execute a predefined fraud detection scenario
 * @param {string} scenarioName - The name of the scenario to run
 * @returns {Promise<Object>} Scenario execution results from the API
 */
export const runScenario = async (scenarioName) => {
  try {
    const response = await apiClient.post(`/scenario/${scenarioName}`);
    return response.data;
  } catch (error) {
    console.error(`[runScenario] Failed to run scenario ${scenarioName}:`, error?.message);
    throw error;
  }
};

/**
 * Reset the blockchain chain state for a specific user
 * @param {string} userId - The user ID whose chain should be reset
 * @returns {Promise<Object>} Reset confirmation from the API
 */
export const resetChain = async (userId) => {
  try {
    const response = await apiClient.post(`/chain/reset`, {
      user_id: userId,
    });
    return response.data;
  } catch (error) {
    console.error(`[resetChain] Failed to reset chain for user ${userId}:`, error?.message);
    throw error;
  }
};

export default apiClient;