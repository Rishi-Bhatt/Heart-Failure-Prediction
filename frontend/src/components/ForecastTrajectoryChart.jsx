/**
 * ForecastTrajectoryChart - A component for visualizing risk forecasts
 *
 * This component creates a scientific visualization of predicted risk trajectories,
 * including confidence intervals and scenario comparisons.
 *
 * References:
 * 1. Cheng, L., et al. (2020). "Temporal Patterns Mining in Electronic Health Records using Deep Learning"
 * 2. Rajkomar, A., et al. (2022). "Machine Learning for Electronic Health Records"
 * 3. Goldstein, B.A., et al. (2021). "Opportunities and Challenges in Developing Risk Prediction Models with Electronic Health Records Data"
 */
import React, { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";
import "chartjs-adapter-date-fns";
import { format } from "date-fns";
import ForecastService from "../services/ForecastService";
import "../styles/ForecastTrajectoryChart.css";

const ForecastTrajectoryChart = ({
  patientId,
  scenarios = [],
  onForecastLoaded = null,
}) => {
  const [forecastData, setForecastData] = useState(null);
  const [forecastHorizon, setForecastHorizon] = useState(6);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const onForecastLoadedRef = useRef(onForecastLoaded);

  // Update the ref when onForecastLoaded changes
  useEffect(() => {
    onForecastLoadedRef.current = onForecastLoaded;
  }, [onForecastLoaded]);

  // Track API request status to prevent duplicate calls
  const [isRequestPending, setIsRequestPending] = useState(false);
  const requestIdRef = useRef(null);
  const lastFetchTimeRef = useRef(0);

  // Function to check if we should throttle a request
  const shouldThrottleRequest = () => {
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchTimeRef.current;

    // Throttle if less than 3 seconds have passed since the last fetch
    if (timeSinceLastFetch < 3000 && lastFetchTimeRef.current !== 0) {
      console.log(
        `Throttling forecast request, only ${timeSinceLastFetch}ms since last fetch`
      );
      return true;
    }

    // Update last fetch time
    lastFetchTimeRef.current = now;
    return false;
  };

  // Fetch forecast data when component mounts or when patientId/forecastHorizon changes
  useEffect(() => {
    // Create a unique request ID to track this specific request
    const currentRequestId = `${patientId}-${forecastHorizon}`;

    // Skip if we're already loading data for this patient and horizon
    if (isRequestPending) {
      console.log(
        "Skipping duplicate forecast request - request already pending"
      );
      return;
    }

    // Skip if we should throttle the request
    if (shouldThrottleRequest()) {
      console.log("Forecast request throttled");
      return;
    }

    requestIdRef.current = currentRequestId;

    const fetchForecast = async () => {
      // Set loading state and mark request as pending
      setIsLoading(true);
      setIsRequestPending(true);
      setError(null);

      try {
        console.log(
          `Fetching forecast for patient ${patientId} with horizon ${forecastHorizon} (request ID: ${currentRequestId})`
        );

        // Use a simple timestamp for cache busting
        const timestamp = new Date().getTime();
        const response = await ForecastService.getPatientForecast(
          patientId,
          forecastHorizon,
          timestamp
        );

        // Check if this request is still the current one
        if (requestIdRef.current !== currentRequestId) {
          console.log("Ignoring response from outdated request");
          return;
        }

        // Handle different response formats
        if (response.data && response.data.status === "success") {
          console.log("Forecast data loaded successfully:", response.data);
          setForecastData(response.data);

          // Call the onForecastLoaded callback if provided
          if (onForecastLoadedRef.current) {
            onForecastLoadedRef.current(response.data);
          }
        } else if (
          response.data &&
          response.data.forecast_values &&
          response.data.forecast_timestamps
        ) {
          // Handle case where response doesn't have status field but has forecast data
          console.log("Forecast data loaded (legacy format):", response.data);

          // Add status field for compatibility
          const enhancedData = {
            ...response.data,
            status: "success",
          };

          setForecastData(enhancedData);

          // Call the onForecastLoaded callback if provided
          if (onForecastLoadedRef.current) {
            onForecastLoadedRef.current(enhancedData);
          }
        } else {
          // Handle error cases
          console.warn(
            "Forecast API returned unexpected format:",
            response.data
          );
          setError("Received unexpected data format from server");
        }
      } catch (err) {
        // Only process error if this is still the current request
        if (requestIdRef.current !== currentRequestId) {
          return;
        }

        console.error("Error fetching forecast:", err);

        // Extract error message from response if available
        let errorMessage = "An error occurred while fetching forecast data";

        if (err.response?.data?.message) {
          errorMessage = err.response.data.message;
        } else if (err.response?.data?.error) {
          errorMessage = err.response.data.error;
        } else if (err.message) {
          errorMessage = err.message;
        }

        setError(errorMessage);
      } finally {
        // Only update loading state if this is still the current request
        if (requestIdRef.current === currentRequestId) {
          setIsLoading(false);
          setIsRequestPending(false);
        }
      }
    };

    if (patientId) {
      // Execute immediately without delay
      fetchForecast();

      // Return cleanup function
      return () => {
        // If there's a pending request when component unmounts, mark it as outdated
        if (requestIdRef.current === currentRequestId) {
          console.log("Cleaning up forecast request on unmount");
          requestIdRef.current = null;
        }
      };
    }
  }, [patientId, forecastHorizon, isRequestPending]);

  // Create or update chart when forecastData or scenarios change
  useEffect(() => {
    if (!forecastData || isLoading) {
      return;
    }

    console.log("Creating forecast chart with data:", {
      forecastData: forecastData ? "present" : "missing",
      scenarios: scenarios.length,
      isLoading,
    });

    // Destroy previous chart if it exists
    if (chartInstance.current) {
      console.log("Destroying previous forecast chart instance");
      chartInstance.current.destroy();
      chartInstance.current = null;
    }

    // Force a small delay to ensure DOM is ready
    setTimeout(() => {
      // Prepare data for the chart
      const createChart = () => {
        if (!chartRef.current) {
          console.error("Chart reference is not available");
          return;
        }

        const ctx = chartRef.current.getContext("2d");
        if (!ctx) {
          console.error("Failed to get 2D context from canvas");
          return;
        }

        // Parse timestamps to Date objects
        const timestamps = forecastData.forecast_timestamps.map(
          (ts) => new Date(ts)
        );
        const values = forecastData.forecast_values;
        const confidenceValues = forecastData.confidence_values;

        // Create datasets array
        const datasets = [
          {
            label: "Predicted Risk",
            data: timestamps.map((date, i) => ({
              x: date,
              y: values[i],
            })),
            borderColor: "rgba(255, 99, 132, 1)",
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            borderWidth: 2,
            pointRadius: 4,
            pointHoverRadius: 6,
            tension: 0.1,
            fill: false,
          },
        ];

        // Add confidence interval dataset
        if (confidenceValues && confidenceValues.length > 0) {
          datasets.push({
            label: "Confidence Interval",
            data: timestamps.map((date, i) => ({
              x: date,
              y: values[i],
              yMin: Math.max(0, values[i] - confidenceValues[i]),
              yMax: Math.min(1, values[i] + confidenceValues[i]),
            })),
            borderColor: "transparent",
            backgroundColor: "rgba(255, 99, 132, 0.2)",
            pointRadius: 0,
            fill: true,
          });
        }

        // Add scenario datasets if available
        scenarios.forEach((scenario, index) => {
          // Use different colors for each scenario
          const colors = [
            {
              border: "rgba(54, 162, 235, 1)",
              background: "rgba(54, 162, 235, 0.2)",
            },
            {
              border: "rgba(75, 192, 192, 1)",
              background: "rgba(75, 192, 192, 0.2)",
            },
            {
              border: "rgba(153, 102, 255, 1)",
              background: "rgba(153, 102, 255, 0.2)",
            },
            {
              border: "rgba(255, 159, 64, 1)",
              background: "rgba(255, 159, 64, 0.2)",
            },
          ];

          const colorIndex = index % colors.length;

          if (scenario.forecast_timestamps && scenario.forecast_values) {
            const scenarioTimestamps = scenario.forecast_timestamps.map(
              (ts) => new Date(ts)
            );

            datasets.push({
              label: `Scenario: ${
                scenario.scenario_name || `Scenario ${index + 1}`
              }`,
              data: scenarioTimestamps.map((date, i) => ({
                x: date,
                y: scenario.forecast_values[i],
              })),
              borderColor: colors[colorIndex].border,
              backgroundColor: colors[colorIndex].background,
              borderWidth: 2,
              pointRadius: 4,
              pointHoverRadius: 6,
              tension: 0.1,
              fill: false,
            });
          }
        });

        // Create the chart
        chartInstance.current = new Chart(ctx, {
          type: "line",
          data: {
            datasets: datasets,
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              tooltip: {
                callbacks: {
                  title: (tooltipItems) => {
                    return format(new Date(tooltipItems[0].raw.x), "PPP");
                  },
                  label: (tooltipItem) => {
                    let label = tooltipItem.dataset.label || "";
                    if (label) {
                      label += ": ";
                    }
                    label += `${(tooltipItem.raw.y * 100).toFixed(1)}%`;
                    return label;
                  },
                  afterLabel: (tooltipItem) => {
                    // Add confidence interval information if available
                    if (
                      tooltipItem.dataset.label === "Confidence Interval" &&
                      tooltipItem.raw.yMin !== undefined &&
                      tooltipItem.raw.yMax !== undefined
                    ) {
                      return `Range: ${(tooltipItem.raw.yMin * 100).toFixed(
                        1
                      )}% - ${(tooltipItem.raw.yMax * 100).toFixed(1)}%`;
                    }
                    return "";
                  },
                },
              },
              title: {
                display: true,
                text: "Risk Trajectory Forecast",
                font: {
                  size: 16,
                  weight: "bold",
                },
              },
              legend: {
                position: "top",
              },
            },
            scales: {
              x: {
                type: "time",
                time: {
                  unit: "month",
                  tooltipFormat: "PPP",
                  displayFormats: {
                    month: "MMM yyyy",
                  },
                },
                title: {
                  display: true,
                  text: "Date",
                  font: {
                    weight: "bold",
                  },
                },
              },
              y: {
                title: {
                  display: true,
                  text: "Risk Score",
                  font: {
                    weight: "bold",
                  },
                },
                min: 0,
                max: 1,
                ticks: {
                  callback: function (value) {
                    return (value * 100).toFixed(0) + "%";
                  },
                },
              },
            },
          },
        });
      };

      // Create the chart
      createChart();
    }, 100); // End of setTimeout

    // Cleanup on unmount or when data changes
    return () => {
      if (chartInstance.current) {
        console.log("Cleaning up forecast chart instance");
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [forecastData, scenarios, isLoading]);

  // Handle forecast horizon change
  const handleHorizonChange = (e) => {
    setForecastHorizon(parseInt(e.target.value));
  };

  // If loading, show loading indicator
  if (isLoading) {
    return (
      <div className="forecast-trajectory-chart">
        <div className="forecast-controls">
          <label>Forecast Horizon:</label>
          <select value={forecastHorizon} onChange={handleHorizonChange}>
            <option value="3">3 Months</option>
            <option value="6">6 Months</option>
            <option value="12">12 Months</option>
          </select>
        </div>
        <div className="loading-indicator">Loading forecast data...</div>
      </div>
    );
  }

  // If error, show error message
  if (error) {
    return (
      <div className="forecast-trajectory-chart">
        <div className="forecast-controls">
          <label>Forecast Horizon:</label>
          <select value={forecastHorizon} onChange={handleHorizonChange}>
            <option value="3">3 Months</option>
            <option value="6">6 Months</option>
            <option value="12">12 Months</option>
          </select>
        </div>
        <div className="error-message">
          <p>Error: {error}</p>
          <p>
            To generate forecasts, please add at least two follow-up visits with
            clinical data.
          </p>
          <div className="error-help">
            <h5>Troubleshooting Steps:</h5>
            <ol>
              <li>Go to the Patient Summary page</li>
              <li>Click on "Add Follow-Up Visit"</li>
              <li>Enter clinical parameters and biomarker values</li>
              <li>Submit the form to record the visit</li>
              <li>Add at least one more follow-up visit</li>
              <li>Return to the Risk Forecast page</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  // If no forecast data, show message
  if (!forecastData) {
    return (
      <div className="forecast-trajectory-chart">
        <div className="forecast-controls">
          <label>Forecast Horizon:</label>
          <select value={forecastHorizon} onChange={handleHorizonChange}>
            <option value="3">3 Months</option>
            <option value="6">6 Months</option>
            <option value="12">12 Months</option>
          </select>
        </div>
        <div className="no-data-message">
          <p>
            No forecast data available. To generate forecasts, please add at
            least two follow-up visits with clinical data.
          </p>
          <div className="error-help">
            <h5>How to Add Visit Data:</h5>
            <ol>
              <li>Go to the Patient Summary page</li>
              <li>Click on "Add Follow-Up Visit"</li>
              <li>Enter clinical parameters and biomarker values</li>
              <li>Submit the form to record the visit</li>
              <li>Add at least one more follow-up visit</li>
              <li>Return to the Risk Forecast page</li>
            </ol>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="forecast-trajectory-chart">
      <div className="forecast-controls">
        <label>Forecast Horizon:</label>
        <select value={forecastHorizon} onChange={handleHorizonChange}>
          <option value="3">3 Months</option>
          <option value="6">6 Months</option>
          <option value="12">12 Months</option>
        </select>
      </div>

      <div className="chart-container">
        <canvas ref={chartRef}></canvas>
      </div>

      <div className="forecast-insights">
        <h4>Forecast Insights</h4>
        <div className="insights-grid">
          <div className="insight-item">
            <span className="insight-label">Current Risk:</span>
            <span className="insight-value">
              {forecastData.current_risk !== null
                ? `${(forecastData.current_risk * 100).toFixed(1)}%`
                : "N/A"}
            </span>
          </div>

          <div className="insight-item">
            <span className="insight-label">Peak Risk:</span>
            <span className="insight-value">
              {`${(forecastData.peak_risk * 100).toFixed(1)}%`}
            </span>
          </div>

          <div className="insight-item">
            <span className="insight-label">Trend:</span>
            <span className="insight-value">
              {forecastData.trend_description}
            </span>
          </div>

          <div className="insight-item">
            <span className="insight-label">3-Month Forecast:</span>
            <span className="insight-value">
              {forecastData.forecast_values.length >= 3
                ? `${(forecastData.forecast_values[2] * 100).toFixed(1)}%`
                : "N/A"}
            </span>
          </div>

          <div className="insight-item">
            <span className="insight-label">6-Month Forecast:</span>
            <span className="insight-value">
              {forecastData.forecast_values.length >= 6
                ? `${(forecastData.forecast_values[5] * 100).toFixed(1)}%`
                : "N/A"}
            </span>
          </div>
        </div>

        {/* Display recommendations from insights */}
        {forecastData.insights && forecastData.insights.recommendations && (
          <div className="recommendations">
            <h5>Recommendations</h5>
            <ul className="recommendations-list">
              {forecastData.insights.recommendations.map(
                (recommendation, index) => (
                  <li key={index} className="recommendation-item">
                    {recommendation}
                  </li>
                )
              )}
            </ul>
          </div>
        )}

        {forecastData.feature_importance &&
          forecastData.feature_importance.length > 0 && (
            <div className="feature-importance">
              <h5>Key Risk Drivers</h5>
              <ul className="importance-list">
                {forecastData.feature_importance
                  .slice(0, 5)
                  .map((feature, index) => (
                    <li key={index} className="importance-item">
                      <span className="feature-name">
                        {feature.name.replace(/_/g, " ")}:
                      </span>
                      <div className="importance-bar-container">
                        <div
                          className="importance-bar"
                          style={{ width: `${feature.importance * 100}%` }}
                        ></div>
                        <span className="importance-value">
                          {(feature.importance * 100).toFixed(1)}%
                        </span>
                      </div>
                    </li>
                  ))}
              </ul>
            </div>
          )}

        <div className="research-notes">
          <h5>Research Notes</h5>
          <p>
            Longitudinal risk forecasting enables proactive intervention
            planning by predicting future risk trajectories. This model
            incorporates temporal patterns in clinical parameters and biomarkers
            to generate personalized risk forecasts.
          </p>
          <p>
            <strong>References:</strong> Cheng, L., et al. (2020). "Temporal
            Patterns Mining in Electronic Health Records using Deep Learning"
          </p>
        </div>
      </div>
    </div>
  );
};

export default ForecastTrajectoryChart;
