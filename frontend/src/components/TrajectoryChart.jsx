import React, { useEffect, useRef, useState } from "react";
import Chart from "chart.js/auto";
import "chartjs-adapter-date-fns";
import { format } from "date-fns";

/**
 * TrajectoryChart - A component for visualizing longitudinal patient data
 *
 * This component creates a scientific visualization of patient trajectories over time,
 * supporting both risk scores and biomarker values with statistical analysis.
 *
 * References:
 * 1. Rizopoulos D. (2012). "Joint Models for Longitudinal and Time-to-Event Data"
 * 2. Diggle P, et al. (2002). "Analysis of Longitudinal Data"
 */
const TrajectoryChart = ({
  trajectoryData,
  analysisData,
  title = "Patient Trajectory",
  yAxisLabel = "Value",
  biomarkerName = null,
}) => {
  // Define all hooks at the top level
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const [renderFailed, setRenderFailed] = useState(false);

  // Reset render failed state when data changes
  useEffect(() => {
    setRenderFailed(false);
  }, [trajectoryData, analysisData]);

  // Main chart creation effect
  useEffect(() => {
    // Force a small delay to ensure DOM is ready
    const timer = setTimeout(() => {
      try {
        // Always log the data we're working with for debugging
        console.log("TrajectoryChart received data:", {
          trajectoryData,
          analysisData,
          title,
          yAxisLabel,
          biomarkerName,
        });

        if (!trajectoryData || trajectoryData.length === 0) {
          console.warn("No trajectory data available");
          return;
        }

        // Destroy previous chart if it exists
        if (chartInstance.current) {
          console.log("Destroying previous chart instance");
          chartInstance.current.destroy();
          chartInstance.current = null;
        }

        // Make sure we have a valid chart reference
        if (!chartRef.current) {
          console.error("Chart reference is not available");
          setRenderFailed(true);
          return;
        }

        // Force a redraw of the canvas to ensure it's ready
        const canvas = chartRef.current;
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        // Log canvas dimensions for debugging
        console.log("Canvas dimensions:", {
          width: canvas.width,
          height: canvas.height,
          offsetWidth: canvas.offsetWidth,
          offsetHeight: canvas.offsetHeight
        });

      console.log("Creating trajectory chart with data:", trajectoryData);

      // Prepare data
      const timestamps = trajectoryData.map((point) => {
        try {
          return new Date(point.timestamp);
        } catch (e) {
          console.warn("Invalid timestamp:", point.timestamp);
          return new Date(); // Default to current date if invalid
        }
      });

      const values = trajectoryData.map((point) => {
        if (point.value === undefined || point.value === null) {
          console.warn("Missing value in trajectory data");
          return 0; // Default value
        }
        return point.value;
      });

      const confidenceValues = trajectoryData.map((point) => point.confidence);

      // Calculate confidence intervals if confidence is available
      const hasConfidence = confidenceValues.some(
        (v) => v !== null && v !== undefined
      );

      // Calculate trend line if we have analysis data
      // Make sure analysisData has all required fields
      const validAnalysisData =
        analysisData &&
        typeof analysisData === "object" &&
        analysisData.trend !== undefined &&
        analysisData.trend !== null &&
        analysisData.intercept !== undefined &&
        analysisData.intercept !== null;

      const hasTrend = validAnalysisData;
      let trendLineData = [];
      let confidenceIntervalData = [];

      if (hasTrend && timestamps.length >= 2) {
        try {
          // Sort timestamps for proper trend line
          const sortedPoints = [...trajectoryData].sort((a, b) => {
            try {
              return new Date(a.timestamp) - new Date(b.timestamp);
            } catch (e) {
              return 0;
            }
          });

          // Get the first timestamp as reference point
          const firstTimestamp = new Date(sortedPoints[0].timestamp);

          // Create trend line data using the actual regression formula
          trendLineData = sortedPoints.map((point) => {
            // Calculate days since first measurement
            const currentDate = new Date(point.timestamp);
            const daysSinceStart =
              (currentDate - firstTimestamp) / (1000 * 60 * 60 * 24);

            // Apply the regression formula: y = slope * x + intercept
            return {
              x: currentDate,
              y: analysisData.trend * daysSinceStart + analysisData.intercept,
            };
          });

          // Add confidence interval if available
          if (
            analysisData.confidence_interval &&
            analysisData.confidence_interval.lower !== undefined &&
            analysisData.confidence_interval.upper !== undefined
          ) {
            // Create confidence interval bands
            confidenceIntervalData = sortedPoints.map((point) => {
              const currentDate = new Date(point.timestamp);
              const daysSinceStart =
                (currentDate - firstTimestamp) / (1000 * 60 * 60 * 24);

              // Calculate lower and upper bounds
              const lowerBound =
                analysisData.confidence_interval.lower * daysSinceStart +
                analysisData.intercept;
              const upperBound =
                analysisData.confidence_interval.upper * daysSinceStart +
                analysisData.intercept;

              return {
                x: currentDate,
                y: analysisData.trend * daysSinceStart + analysisData.intercept,
                yMin: lowerBound,
                yMax: upperBound,
              };
            });
          }
        } catch (e) {
          console.warn("Error creating trend line:", e);
        }
      }

      // Define variables for chart creation
      let ctx = null;
      let datasets = [];

      // Create the chart
      try {
        if (chartRef.current) {
          ctx = chartRef.current.getContext("2d");
          if (!ctx) {
            console.error("Failed to get 2D context from canvas");
            setRenderFailed(true);
            return;
          }
        } else {
          console.error("Chart reference is not available");
          setRenderFailed(true);
          return;
        }

        // Determine color based on biomarker or risk
        let borderColor, backgroundColor;
        if (biomarkerName === "nt_probnp") {
          borderColor = "rgba(153, 102, 255, 1)";
          backgroundColor = "rgba(153, 102, 255, 0.2)";
        } else if (biomarkerName) {
          borderColor = "rgba(75, 192, 192, 1)";
          backgroundColor = "rgba(75, 192, 192, 0.2)";
        } else {
          // Risk trajectory
          borderColor = "rgba(255, 99, 132, 1)";
          backgroundColor = "rgba(255, 99, 132, 0.2)";
        }

        // Create datasets array
        datasets = [
          {
            label: biomarkerName ? `${biomarkerName} Values` : "Risk Score",
            data: trajectoryData.map((point) => {
              try {
                return {
                  x: new Date(point.timestamp),
                  y:
                    point.value !== undefined && point.value !== null
                      ? point.value
                      : 0,
                };
              } catch (e) {
                console.warn("Error creating data point:", e);
                return {
                  x: new Date(),
                  y: 0,
                };
              }
            }),
            borderColor: borderColor,
            backgroundColor: backgroundColor,
            borderWidth: 2,
            pointRadius: 5,
            pointHoverRadius: 7,
            tension: 0.1,
          },

          // Add outliers as separate dataset if available
          ...(analysisData &&
          analysisData.outliers &&
          analysisData.outliers.length > 0
            ? [
                {
                  label: "Outliers",
                  data: analysisData.outliers.map((outlier) => ({
                    x: new Date(outlier.timestamp),
                    y: outlier.value,
                  })),
                  borderColor: "rgba(255, 159, 64, 1)",
                  backgroundColor: "rgba(255, 159, 64, 0.7)",
                  borderWidth: 2,
                  pointRadius: 6,
                  pointStyle: "triangle",
                  pointHoverRadius: 8,
                  showLine: false,
                },
              ]
            : []),
        ];

        // Add trend line if available
        if (hasTrend && trendLineData.length > 0) {
          try {
            // Add confidence interval band if available
            if (confidenceIntervalData.length > 0) {
              datasets.push({
                label: "95% Confidence Interval",
                data: confidenceIntervalData,
                borderColor: "transparent",
                backgroundColor: "rgba(54, 162, 235, 0.2)",
                pointRadius: 0,
                fill: false,
                tension: 0,
              });
            }

            // Add trend line
            const trendLabel = biomarkerName
              ? `Trend (${analysisData.trend > 0 ? "+" : ""}${
                  analysisData.trend_monthly?.toFixed(2) ||
                  analysisData.trend.toFixed(2)
                }/month)`
              : `Trend (${analysisData.trend > 0 ? "+" : ""}${(
                  (analysisData.trend_monthly || analysisData.trend * 30) * 100
                ).toFixed(1)}%/month)`;

            datasets.push({
              label: trendLabel,
              data: trendLineData,
              borderColor: "rgba(54, 162, 235, 1)",
              borderWidth: 2,
              borderDash: [5, 5],
              pointRadius: 0,
              tension: 0,
              fill: false,
            });

            // Add significance indicator to trend line label
            if (
              analysisData.p_value !== null &&
              analysisData.p_value !== undefined
            ) {
              const lastIndex = datasets.length - 1;
              if (analysisData.p_value < 0.05) {
                datasets[
                  lastIndex
                ].label += ` (p=${analysisData.p_value.toFixed(3)}*)`;
              } else {
                datasets[
                  lastIndex
                ].label += ` (p=${analysisData.p_value.toFixed(3)})`;
              }
            }
          } catch (e) {
            console.warn("Error adding trend line:", e);
          }
        }

        // Add confidence intervals if available
        if (hasConfidence) {
          try {
            datasets.push({
              label: "Confidence Interval",
              data: trajectoryData.map((point) => {
                try {
                  return {
                    x: new Date(point.timestamp),
                    y:
                      point.value !== undefined && point.value !== null
                        ? point.value
                        : 0,
                    yMin: point.confidence
                      ? Math.max(0, point.value - point.confidence * 0.5)
                      : null,
                    yMax: point.confidence
                      ? Math.min(1, point.value + point.confidence * 0.5)
                      : null,
                  };
                } catch (e) {
                  console.warn("Error creating confidence interval:", e);
                  return {
                    x: new Date(),
                    y: 0,
                    yMin: null,
                    yMax: null,
                  };
                }
              }),
              borderColor: "transparent",
              backgroundColor: "rgba(54, 162, 235, 0.3)",
              pointRadius: 0,
              type: "line",
            });
          } catch (e) {
            console.warn("Error adding confidence intervals:", e);
          }
        }
      } catch (e) {
        console.error("Error setting up chart data:", e);
      }

      try {
        // Make sure datasets is defined
        if (!datasets || !Array.isArray(datasets) || datasets.length === 0) {
          console.error("No valid datasets for chart");
          return;
        }

        console.log("Creating chart with datasets:", datasets);

        if (!ctx) {
          console.error("Canvas context is not available");
          return;
        }

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
                    try {
                      if (
                        !tooltipItems ||
                        !tooltipItems[0] ||
                        !tooltipItems[0].raw ||
                        !tooltipItems[0].raw.x
                      ) {
                        return "Unknown Date";
                      }
                      return format(new Date(tooltipItems[0].raw.x), "PPP p");
                    } catch (e) {
                      console.warn("Error formatting tooltip title:", e);
                      return "Unknown Date";
                    }
                  },
                  label: (tooltipItem) => {
                    try {
                      let label = tooltipItem.dataset.label || "";
                      if (label) {
                        label += ": ";
                      }

                      if (
                        !tooltipItem.raw ||
                        tooltipItem.raw.y === undefined ||
                        tooltipItem.raw.y === null
                      ) {
                        return `${label}N/A`;
                      }

                      if (biomarkerName === "nt_probnp") {
                        label += `${tooltipItem.raw.y.toFixed(0)} pg/mL`;
                      } else if (biomarkerName) {
                        label += tooltipItem.raw.y.toFixed(2);
                      } else {
                        label += `${(tooltipItem.raw.y * 100).toFixed(1)}%`;
                      }

                      return label;
                    } catch (e) {
                      console.warn("Error formatting tooltip label:", e);
                      return "Error displaying value";
                    }
                  },
                  afterLabel: (tooltipItem) => {
                    try {
                      // Handle outlier tooltips
                      if (
                        tooltipItem.dataset.label === "Outliers" &&
                        analysisData &&
                        analysisData.outliers
                      ) {
                        const outlier =
                          analysisData.outliers[tooltipItem.dataIndex];
                        if (outlier) {
                          return [
                            `Z-score: ${outlier.z_score.toFixed(2)}`,
                            `This value deviates significantly from the trend.`,
                          ];
                        }
                        return "";
                      }

                      // Handle confidence tooltips for regular data points
                      if (
                        !trajectoryData ||
                        !tooltipItem ||
                        tooltipItem.dataIndex === undefined ||
                        tooltipItem.dataset.label !==
                          (biomarkerName
                            ? `${biomarkerName} Values`
                            : "Risk Score")
                      ) {
                        return "";
                      }

                      const dataPoint = trajectoryData[tooltipItem.dataIndex];
                      if (dataPoint && dataPoint.confidence) {
                        return `Confidence: ±${(
                          (dataPoint.confidence * 100) /
                          2
                        ).toFixed(1)}%`;
                      }
                      return "";
                    } catch (e) {
                      console.warn("Error formatting tooltip afterLabel:", e);
                      return "";
                    }
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
      } catch (e) {
        console.error("Error creating chart:", e);
        setRenderFailed(true);
      }
    } catch (e) {
      console.error("Error in chart effect:", e);
      setRenderFailed(true);
    }

      }, 100); // Small delay to ensure DOM is ready

      // Cleanup function to destroy chart when component unmounts or data changes
      return () => {
        clearTimeout(timer);
        if (chartInstance.current) {
          console.log("Cleaning up chart instance on unmount or data change");
          chartInstance.current.destroy();
          chartInstance.current = null;
        }
      };
  }, [trajectoryData, analysisData, title, yAxisLabel, biomarkerName]);

  // Function to render analysis section
  const renderAnalysisSection = () => {
    if (!analysisData) return null;

    return (
      <div
        className="trajectory-analysis"
        style={{ marginTop: "15px", fontSize: "0.9rem", color: "#666" }}
      >
        <h4 style={{ fontSize: "1rem", marginBottom: "10px" }}>
          Statistical Analysis
        </h4>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "20px" }}>
          <div style={{ flex: "1", minWidth: "250px" }}>
            <h5 style={{ fontSize: "0.95rem", marginBottom: "5px" }}>
              Summary Statistics
            </h5>
            <p>
              <strong>Data points:</strong> {analysisData?.count || 0}
              <br />
              <strong>Mean value:</strong>{" "}
              {biomarkerName
                ? analysisData?.mean?.toFixed(2)
                : `${(analysisData?.mean * 100).toFixed(1)}%`}
              <br />
              {analysisData?.median !== undefined && (
                <>
                  <strong>Median value:</strong>{" "}
                  {biomarkerName
                    ? analysisData?.median?.toFixed(2)
                    : `${(analysisData?.median * 100).toFixed(1)}%`}
                  <br />
                </>
              )}
              <strong>Range:</strong>{" "}
              {biomarkerName
                ? `${analysisData?.min?.toFixed(
                    2
                  )} - ${analysisData?.max?.toFixed(2)}`
                : `${(analysisData?.min * 100).toFixed(1)}% - ${(
                    analysisData?.max * 100
                  ).toFixed(1)}%`}
              <br />
              {analysisData?.std_dev !== undefined && (
                <>
                  <strong>Standard deviation:</strong>{" "}
                  {biomarkerName
                    ? analysisData?.std_dev?.toFixed(2)
                    : `${(analysisData?.std_dev * 100).toFixed(1)}%`}
                </>
              )}
            </p>
          </div>

          {analysisData?.trend !== null &&
            analysisData?.trend !== undefined && (
              <div style={{ flex: "1", minWidth: "250px" }}>
                <h5 style={{ fontSize: "0.95rem", marginBottom: "5px" }}>
                  Trend Analysis
                </h5>
                <p>
                  <strong>Trend (daily):</strong>{" "}
                  {`${analysisData.trend > 0 ? "+" : ""}${
                    biomarkerName
                      ? analysisData.trend.toFixed(4)
                      : (analysisData.trend * 100).toFixed(3) + "%"
                  } per day`}
                  <br />
                  {analysisData?.trend_monthly !== undefined && (
                    <>
                      <strong>Trend (monthly):</strong>{" "}
                      {`${analysisData.trend_monthly > 0 ? "+" : ""}${
                        biomarkerName
                          ? analysisData.trend_monthly.toFixed(2)
                          : (analysisData.trend_monthly * 100).toFixed(1) + "%"
                      } per month`}
                      <br />
                    </>
                  )}
                  {analysisData?.p_value !== undefined && (
                    <>
                      <strong>p-value:</strong>{" "}
                      {analysisData.p_value.toFixed(3)}
                      {analysisData.p_value < 0.05 &&
                        " (statistically significant)"}
                      <br />
                    </>
                  )}
                  {analysisData?.r_squared !== undefined && (
                    <>
                      <strong>R²:</strong> {analysisData.r_squared.toFixed(3)}
                      <br />
                    </>
                  )}
                  {analysisData?.confidence_interval && (
                    <>
                      <strong>95% CI (slope):</strong>{" "}
                      {biomarkerName
                        ? `${analysisData.confidence_interval.lower.toFixed(
                            4
                          )} to ${analysisData.confidence_interval.upper.toFixed(
                            4
                          )}`
                        : `${(
                            analysisData.confidence_interval.lower * 100
                          ).toFixed(3)}% to ${(
                            analysisData.confidence_interval.upper * 100
                          ).toFixed(3)}%`}
                      <br />
                    </>
                  )}
                  {analysisData?.clinically_significant !== undefined && (
                    <>
                      <strong>Clinical significance:</strong>{" "}
                      {analysisData.clinically_significant ? (
                        <span style={{ color: "#4caf50" }}>Yes</span>
                      ) : (
                        <span style={{ color: "#ff9800" }}>No</span>
                      )}
                    </>
                  )}
                </p>
              </div>
            )}
        </div>

        {analysisData?.outliers && analysisData.outliers.length > 0 && (
          <div style={{ marginTop: "10px" }}>
            <h5 style={{ fontSize: "0.95rem", marginBottom: "5px" }}>
              Outliers Detected
            </h5>
            <p>
              {analysisData.outliers.length} outlier
              {analysisData.outliers.length !== 1 ? "s" : ""} detected. These
              points deviate significantly from the overall trend and may
              represent important clinical events.
            </p>
          </div>
        )}

        {analysisData?.message && (
          <div style={{ marginTop: "10px", fontWeight: "bold" }}>
            <strong>Interpretation:</strong> {analysisData.message}
          </div>
        )}

        <div style={{ marginTop: "10px", fontSize: "0.8rem" }}>
          <em>* p &lt; 0.05 indicates statistical significance</em>
        </div>
      </div>
    );
  };

  // Render no data message if no trajectory data
  if (!trajectoryData || trajectoryData.length === 0) {
    return (
      <div className="no-data-message">
        <p>
          No trajectory data available. Add follow-up visits to see patient
          progression over time.
        </p>
      </div>
    );
  }

  // Render error fallback if chart rendering failed
  if (renderFailed) {
    return (
      <div className="trajectory-chart-container">
        <div
          style={{
            height: "400px",
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexDirection: "column",
            backgroundColor: "#f8f9fa",
            border: "1px dashed #ccc",
            borderRadius: "4px",
            padding: "20px",
          }}
        >
          <h4>Chart Rendering Issue</h4>
          <p>
            There was a problem rendering the chart. Here's a summary of the
            data:
          </p>
          <ul style={{ textAlign: "left" }}>
            <li>Data points: {trajectoryData?.length || 0}</li>
            <li>
              Time range:{" "}
              {trajectoryData && trajectoryData.length > 0
                ? `${new Date(
                    trajectoryData[0].timestamp
                  ).toLocaleDateString()} to ${new Date(
                    trajectoryData[trajectoryData.length - 1].timestamp
                  ).toLocaleDateString()}`
                : "N/A"}
            </li>
            {analysisData?.trend && (
              <li>
                Trend: {analysisData.trend > 0 ? "+" : ""}
                {biomarkerName
                  ? analysisData.trend.toFixed(2)
                  : `${(analysisData.trend * 100).toFixed(1)}%`}{" "}
                per visit
              </li>
            )}
          </ul>
        </div>
        {renderAnalysisSection()}
      </div>
    );
  }

  // Render the chart
  return (
    <div className="trajectory-chart-container">
      <div style={{ height: "400px", width: "100%" }}>
        <canvas ref={chartRef} onError={() => setRenderFailed(true)}></canvas>
      </div>
      {renderAnalysisSection()}
    </div>
  );
};

export default TrajectoryChart;
