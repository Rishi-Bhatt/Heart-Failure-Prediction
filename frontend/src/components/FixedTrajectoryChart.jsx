import React, { useState, useEffect, useRef } from "react";
import { Chart, registerables } from "chart.js";
import { format } from "date-fns";
import "./FixedCharts.css";

// Register all Chart.js components
Chart.register(...registerables);

/**
 * TrajectoryChart - A component for visualizing patient trajectory data
 *
 * This is a simplified version that focuses on reliable rendering
 */
const FixedTrajectoryChart = ({
  trajectoryData,
  analysisData,
  title,
  yAxisLabel,
  biomarkerName,
}) => {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [renderFailed, setRenderFailed] = useState(false);

  // Reset render failed state when data changes
  useEffect(() => {
    setRenderFailed(false);
  }, [trajectoryData, analysisData]);

  // Create chart when component mounts or data changes
  useEffect(() => {
    // Don't try to render if we don't have data
    if (!trajectoryData || trajectoryData.length === 0) {
      console.log("No trajectory data available");
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
          setRenderFailed(true);
          return;
        }

        // Get the canvas context
        const ctx = chartRef.current.getContext("2d");
        if (!ctx) {
          console.error("Failed to get 2D context from canvas");
          setRenderFailed(true);
          return;
        }

        // Prepare data for the chart
        const datasets = [];

        // Main dataset with patient values
        datasets.push({
          label: biomarkerName ? `${biomarkerName} Values` : "Risk Score",
          data: trajectoryData.map((point) => ({
            x: new Date(point.timestamp || point.date),
            y: point.value,
          })),
          borderColor: biomarkerName
            ? "rgba(153, 102, 255, 1)"
            : "rgba(255, 99, 132, 1)",
          backgroundColor: biomarkerName
            ? "rgba(153, 102, 255, 0.2)"
            : "rgba(255, 99, 132, 0.2)",
          borderWidth: 2,
          pointRadius: 5,
          pointHoverRadius: 7,
          tension: 0.1,
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

                    if (biomarkerName === "nt_probnp") {
                      label += `${tooltipItem.raw.y.toFixed(0)} pg/mL`;
                    } else if (biomarkerName) {
                      label += tooltipItem.raw.y.toFixed(2);
                    } else {
                      label += `${(tooltipItem.raw.y * 100).toFixed(1)}%`;
                    }

                    return label;
                  },
                },
              },
              title: {
                display: true,
                text: title,
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
                  unit: "day",
                  tooltipFormat: "PPP",
                  displayFormats: {
                    day: "MMM d",
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
                  text: yAxisLabel,
                  font: {
                    weight: "bold",
                  },
                },
                min: biomarkerName ? undefined : 0,
                max: biomarkerName ? undefined : 1,
                ticks: {
                  callback: function (value) {
                    if (!biomarkerName) {
                      return (value * 100).toFixed(0) + "%";
                    }
                    return value;
                  },
                },
              },
            },
          },
        });

        console.log("Chart created successfully");
      } catch (error) {
        console.error("Error creating chart:", error);
        setRenderFailed(true);
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
  }, [trajectoryData, analysisData, title, yAxisLabel, biomarkerName]);

  // If rendering failed, show an error message
  if (renderFailed) {
    return (
      <div className="trajectory-chart error">
        <div className="error-message">
          <p>Failed to render chart. Please try refreshing the page.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="trajectory-chart">
      <div className="chart-container">
        <canvas ref={chartRef}></canvas>
      </div>
    </div>
  );
};

export default FixedTrajectoryChart;
