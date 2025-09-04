import React, { useState, useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";
import { format } from "date-fns";
import ForecastService from "../services/ForecastService";
import "./FixedCharts.css";

// Register all Chart.js components
Chart.register(...registerables);

/**
 * FixedForecastChart - A simplified component for visualizing forecast data
 *
 * This is a simplified version that focuses on reliable rendering
 */
const FixedForecastChart = ({ patientId, onForecastLoaded }) => {
  const [forecastData, setForecastData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecastHorizon, setForecastHorizon] = useState(6);

  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const onForecastLoadedRef = useRef(onForecastLoaded);

  // Update the ref when the callback changes
  useEffect(() => {
    onForecastLoadedRef.current = onForecastLoaded;
  }, [onForecastLoaded]);

  // Fetch forecast data when component mounts or when patientId/forecastHorizon changes
  useEffect(() => {
    const fetchForecast = async () => {
      if (!patientId) return;

      setIsLoading(true);
      setError(null);

      try {
        console.log(
          `Fetching forecast for patient ${patientId} with horizon ${forecastHorizon}`
        );

        // Use a timestamp to prevent caching
        const timestamp = new Date().getTime();

        try {
          const response = await ForecastService.getPatientForecast(
            patientId,
            forecastHorizon,
            timestamp
          );

          if (response.data) {
            console.log("Forecast data loaded:", response.data);

            // Handle different response formats
            if (response.data.status === "success") {
              setForecastData(response.data);
            } else if (
              response.data.forecast_values &&
              response.data.forecast_timestamps
            ) {
              // Legacy format without status field
              const enhancedData = {
                ...response.data,
                status: "success",
              };
              setForecastData(enhancedData);
            } else {
              setError("Received unexpected data format from server");
            }

            // Call the onForecastLoaded callback if provided
            if (onForecastLoadedRef.current && response.data) {
              onForecastLoadedRef.current(response.data);
            }
          } else {
            setError("No data received from server");
          }
        } catch (innerErr) {
          console.error("Error processing forecast data:", innerErr);
          setError("Error processing forecast data. Please try again.");
        }
      } catch (err) {
        console.error("Error fetching forecast:", err);
        setError("Failed to load forecast data. Please try again later.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchForecast();
  }, [patientId, forecastHorizon]);

  // Create chart when forecast data is available
  useEffect(() => {
    if (!forecastData || isLoading) {
      return;
    }

    // Use a timeout to ensure DOM is ready
    const timer = setTimeout(() => {
      try {
        // Clean up any existing chart
        if (chartInstance.current) {
          chartInstance.current.destroy();
          chartInstance.current = null;
        }

        // Make sure we have a canvas to draw on
        if (!chartRef.current) {
          console.error("Chart reference is not available");
          return;
        }

        // Get the canvas context
        const ctx = chartRef.current.getContext("2d");
        if (!ctx) {
          console.error("Failed to get 2D context from canvas");
          return;
        }

        // Parse timestamps to Date objects
        const timestamps = (forecastData.forecast_timestamps || []).map(
          (ts) => new Date(ts)
        );
        const values = forecastData.forecast_values || [];

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

        console.log("Forecast chart created successfully");
      } catch (error) {
        console.error("Error creating forecast chart:", error);
      }
    }, 100);

    // Clean up function
    return () => {
      clearTimeout(timer);
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [forecastData, isLoading]);

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
              {forecastData.peak_risk
                ? `${(forecastData.peak_risk * 100).toFixed(1)}%`
                : "N/A"}
            </span>
          </div>

          <div className="insight-item">
            <span className="insight-label">Trend:</span>
            <span className="insight-value">
              {forecastData.trend_description || "Stable"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FixedForecastChart;
