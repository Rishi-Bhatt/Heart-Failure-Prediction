/**
 * ForecastService - Service for interacting with the forecasting API
 *
 * This service provides methods for retrieving forecasts and generating
 * scenario-based forecasts for patients.
 */
import axios from "axios";

const API_BASE_URL = "http://localhost:8083";

export default {
  /**
   * Get a forecast for a patient
   *
   * @param {string} patientId - The ID of the patient
   * @param {number} horizon - The forecast horizon in months (default: 6)
   * @param {string} cacheKey - Optional cache key to prevent duplicate requests
   * @returns {Promise} - Promise resolving to forecast data
   */
  getPatientForecast(patientId, horizon = 6, cacheKey = null) {
    // Use provided cacheKey or generate a timestamp
    const cacheBuster = cacheKey || new Date().getTime();

    // Use a consistent parameter name for cache busting
    return axios.get(
      `${API_BASE_URL}/api/patients/${patientId}/forecast?horizon=${horizon}&cache=${cacheBuster}`,
      {
        // Add headers to prevent browser caching
        headers: {
          "Cache-Control": "no-cache",
          Pragma: "no-cache",
          Expires: "0",
        },
      }
    );
  },

  /**
   * Generate a scenario-based forecast for a patient
   *
   * @param {string} patientId - The ID of the patient
   * @param {object} scenarioParams - The scenario parameters
   * @param {number} horizon - The forecast horizon in months (default: 6)
   * @param {string} cacheKey - Optional cache key to prevent duplicate requests
   * @returns {Promise} - Promise resolving to forecast data
   */
  getScenarioForecast(patientId, scenarioParams, horizon = 6, cacheKey = null) {
    // Use provided cacheKey or generate a timestamp
    const cacheBuster = cacheKey || new Date().getTime();

    return axios.post(
      `${API_BASE_URL}/api/patients/${patientId}/forecast/scenario?horizon=${horizon}&cache=${cacheBuster}`,
      scenarioParams,
      {
        // Add headers to prevent browser caching
        headers: {
          "Cache-Control": "no-cache",
          Pragma: "no-cache",
          Expires: "0",
        },
      }
    );
  },

  /**
   * Get a specific forecast by ID
   *
   * @param {string} forecastId - The ID of the forecast
   * @returns {Promise} - Promise resolving to forecast data
   */
  getForecast(forecastId) {
    const timestamp = new Date().getTime(); // Add timestamp to prevent caching
    return axios.get(
      `${API_BASE_URL}/api/forecasts/${forecastId}?t=${timestamp}`
    );
  },

  /**
   * Get all forecasts for a patient
   *
   * @param {string} patientId - The ID of the patient
   * @returns {Promise} - Promise resolving to forecast data
   */
  getPatientForecasts(patientId) {
    const timestamp = new Date().getTime(); // Add timestamp to prevent caching
    return axios.get(
      `${API_BASE_URL}/api/forecasts?patient_id=${patientId}&t=${timestamp}`
    );
  },

  /**
   * Train the forecasting model
   *
   * @param {number} forecastHorizon - The forecast horizon in months (default: 6)
   * @returns {Promise} - Promise resolving to training results
   */
  trainForecastModel(forecastHorizon = 6) {
    const timestamp = new Date().getTime(); // Add timestamp to prevent caching
    return axios.post(`${API_BASE_URL}/api/forecast/train?t=${timestamp}`, {
      forecast_horizon: forecastHorizon,
    });
  },
};
