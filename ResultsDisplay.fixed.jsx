import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Chart from "chart.js/auto";
import axios from "axios";
import BiomarkerImpactChart from "./BiomarkerImpactChart";
import "../styles/abnormality-timeline.css";
import "../styles/spinner.css";

const ResultsDisplay = ({ predictionResult }) => {
  const navigate = useNavigate();
  const ecgChartRef = useRef(null);
  const shapChartRef = useRef(null);
  const [patientVerified, setPatientVerified] = useState(false);
  const [verificationAttempts, setVerificationAttempts] = useState(0);
  const [patientError, setPatientError] = useState(null);
  const MAX_VERIFICATION_ATTEMPTS = 10; // Increase max attempts

  // Redirect to home if no prediction result
  useEffect(() => {
    if (!predictionResult) {
      navigate("/");
      return;
    }

    // Verify that the patient was actually saved
    const verifyPatient = async () => {
      try {
        if (!predictionResult.patient_id) {
          console.error("No patient ID in prediction result");
          setPatientError("No patient ID found in prediction result");
          return;
        }

        console.log(
          `Verifying patient with ID: ${predictionResult.patient_id}`
        );
        
        // Try to get the patient data directly instead of using the check endpoint
        try {
          const response = await axios.get(
            `http://localhost:8082/api/patients/${predictionResult.patient_id}`,
            {
              headers: {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                Pragma: "no-cache",
                Expires: "0",
              },
            }
          );

          console.log("Patient verification response:", response.data);

          if (response.data && response.data.patient_id) {
            console.log("Patient verified successfully!");
            setPatientVerified(true);
            return;
          }
        } catch (getError) {
          console.error("Error getting patient data:", getError);
          // Continue with the check endpoint as fallback
        }
        
        // Fallback to the check endpoint
        const checkResponse = await axios.get(
          `http://localhost:8082/api/patients/check/${predictionResult.patient_id}`,
          {
            headers: {
              "Cache-Control": "no-cache, no-store, must-revalidate",
              Pragma: "no-cache",
              Expires: "0",
            },
          }
        );

        console.log("Patient check response:", checkResponse.data);

        if (checkResponse.data.exists) {
          console.log("Patient verified successfully!");
          setPatientVerified(true);
        } else {
          console.error("Patient file does not exist");
          setPatientError("Patient file does not exist. Please try again.");

          // Try again if we haven't tried too many times
          if (verificationAttempts < MAX_VERIFICATION_ATTEMPTS) {
            const nextAttempt = verificationAttempts + 1;
            setVerificationAttempts(nextAttempt);

            // Increase delay with each attempt (exponential backoff)
            const delay = Math.min(1000 * Math.pow(1.5, nextAttempt), 10000);
            console.log(
              `Scheduling retry attempt ${nextAttempt}/${MAX_VERIFICATION_ATTEMPTS} in ${delay}ms`
            );

            setTimeout(verifyPatient, delay);
          }
        }
      } catch (error) {
        console.error("Error verifying patient:", error);
        setPatientError("Error verifying patient. Please try again.");
        
        // Try again if we haven't tried too many times
        if (verificationAttempts < MAX_VERIFICATION_ATTEMPTS) {
          const nextAttempt = verificationAttempts + 1;
          setVerificationAttempts(nextAttempt);

          // Increase delay with each attempt (exponential backoff)
          const delay = Math.min(1000 * Math.pow(1.5, nextAttempt), 10000);
          console.log(
            `Scheduling retry attempt ${nextAttempt}/${MAX_VERIFICATION_ATTEMPTS} in ${delay}ms`
          );

          setTimeout(verifyPatient, delay);
        }
      }
    };

    // Start verification process
    verifyPatient();
  }, [predictionResult, navigate, verificationAttempts]);

  // Create ECG chart
  useEffect(() => {
    if (predictionResult && ecgChartRef.current) {
      const ecgCtx = ecgChartRef.current.getContext("2d");

      // Destroy previous chart if it exists
      if (ecgChartRef.current.chart) {
        ecgChartRef.current.chart.destroy();
      }

      // Log ECG data for debugging
      console.log(
        "ECG Signal Data:",
        predictionResult.ecg_signal
          ? predictionResult.ecg_signal.slice(0, 10)
          : "No ECG data"
      );
      console.log(
        "ECG Time Data:",
        predictionResult.ecg_time
          ? predictionResult.ecg_time.slice(0, 10)
          : "No time data"
      );

      // Ensure we have valid ECG data
      if (
        !predictionResult.ecg_signal ||
        !predictionResult.ecg_time ||
        predictionResult.ecg_signal.length === 0 ||
        predictionResult.ecg_time.length === 0
      ) {
        console.error("Invalid ECG data received");
        return;
      }

      // Create abnormality regions
      const abnormalityRegions = [];

      // Add regions for each abnormality type
      Object.entries(predictionResult.abnormalities).forEach(
        ([type, instances]) => {
          if (instances.length > 0) {
            instances.forEach((instance) => {
              const startTime = instance.time;
              const endTime = instance.time + instance.duration;

              let color;
              switch (type) {
                case "PVCs":
                  color = "rgba(255, 99, 132, 0.3)";
                  break;
                case "Flatlines":
                  color = "rgba(54, 162, 235, 0.3)";
                  break;
                case "Tachycardia":
                  color = "rgba(255, 206, 86, 0.3)";
                  break;
                case "Bradycardia":
                  color = "rgba(75, 192, 192, 0.3)";
                  break;
                case "QT_prolongation":
                  color = "rgba(153, 102, 255, 0.3)";
                  break;
                case "Atrial_Fibrillation":
                  color = "rgba(255, 159, 64, 0.3)";
                  break;
                default:
                  color = "rgba(201, 203, 207, 0.3)";
              }

              abnormalityRegions.push({
                type: "box",
                xMin: startTime,
                xMax: endTime,
                backgroundColor: color,
                borderColor: color.replace("0.3", "0.8"),
                borderWidth: 1,
              });
            });
          }
        }
      );

      // Create ECG chart
      const ecgChart = new Chart(ecgCtx, {
        type: "line",
        data: {
          labels: predictionResult.ecg_time,
          datasets: [
            {
              label: "ECG Signal",
              data: predictionResult.ecg_signal,
              borderColor: "rgb(75, 192, 192)",
              borderWidth: 1.5,
              pointRadius: 0,
              tension: 0.1,
              fill: false,
              cubicInterpolationMode: "monotone",
            },
          ],
        },
        // Chart options
        options: {
          animation: {
            duration: 1000, // Animate for 1 second when data changes
          },
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: {
                display: true,
                text: "Time (seconds)",
              },
              ticks: {
                maxTicksLimit: 10,
                callback: function (value) {
                  return value.toFixed(1);
                },
              },
            },
            y: {
              title: {
                display: true,
                text: "Amplitude",
              },
              min: -1.5, // Set min value to ensure consistent scale
              max: 1.5, // Set max value to ensure consistent scale
              ticks: {
                stepSize: 0.5,
              },
            },
          },
          plugins: {
            annotation: {
              annotations: abnormalityRegions,
            },
            title: {
              display: true,
              text: "ECG Signal with Abnormality Regions",
            },
            tooltip: {
              callbacks: {
                label: function (context) {
                  return `Amplitude: ${context.raw.toFixed(3)}`;
                },
              },
            },
          },
        },
      });

      // Save chart instance for cleanup
      ecgChartRef.current.chart = ecgChart;
    }
  }, [predictionResult]);

  // Create SHAP chart
  useEffect(() => {
    if (
      predictionResult &&
      shapChartRef.current &&
      predictionResult.shap_values
    ) {
      const shapCtx = shapChartRef.current.getContext("2d");

      // Destroy previous chart if it exists
      if (shapChartRef.current.chart) {
        shapChartRef.current.chart.destroy();
      }

      // Get top 10 features by absolute SHAP value
      const shapData = predictionResult.shap_values;
      const featureImportance = shapData.feature_names.map((name, index) => ({
        name: name,
        value: shapData.values[index],
      }));

      // Sort by absolute value and take top 10
      const topFeatures = featureImportance
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 10);

      // Create colors based on value (positive = red, negative = blue)
      const backgroundColors = topFeatures.map((feature) =>
        feature.value >= 0
          ? "rgba(255, 99, 132, 0.7)"
          : "rgba(54, 162, 235, 0.7)"
      );

      // Create SHAP chart
      const shapChart = new Chart(shapCtx, {
        type: "bar",
        data: {
          labels: topFeatures.map((f) => {
            // Make feature names more readable
            const nameMap = {
              age: "Age",
              gender: "Gender",
              blood_pressure: "Blood Pressure",
              cholesterol: "Cholesterol",
              fasting_blood_sugar: "Blood Sugar",
              max_heart_rate: "Max Heart Rate",
              exercise_induced_angina: "Exercise Angina",
              st_depression: "ST Depression",
              slope_of_st: "ST Slope",
              number_of_major_vessels: "Major Vessels",
              thalassemia: "Thalassemia",
              prior_cardiac_event: "Prior Cardiac Event",
              nt_probnp: "NT-proBNP Biomarker",
            };
            return nameMap[f.name] || f.name;
          }),
          datasets: [
            {
              label: "Feature Importance (SHAP value)",
              data: topFeatures.map((f) => f.value),
              backgroundColor: backgroundColors,
              borderColor: backgroundColors.map((c) => c.replace("0.7", "1")),
              borderWidth: 1,
            },
          ],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            title: {
              display: true,
              text: "Feature Importance (SHAP Values)",
            },
            tooltip: {
              callbacks: {
                label: function (context) {
                  const value = context.raw;
                  return `Impact: ${value.toFixed(4)} ${
                    value >= 0 ? "(increases risk)" : "(decreases risk)"
                  }`;
                },
              },
            },
          },
        },
      });

      // Save chart instance for cleanup
      shapChartRef.current.chart = shapChart;
    }
  }, [predictionResult]);

  if (!predictionResult) {
    return <div>Loading...</div>;
  }

  // Determine risk level
  const riskLevel =
    predictionResult.prediction >= 0.7
      ? "High"
      : predictionResult.prediction >= 0.3
      ? "Medium"
      : "Low";

  const riskClass = `risk-${riskLevel.toLowerCase()}`;

  // Function to get color for abnormality type
  const getAbnormalityColor = (type) => {
    const colorMap = {
      PVCs: "#ff5252",
      Flatlines: "#9c27b0",
      Tachycardia: "#ff9800",
      Bradycardia: "#2196f3",
      QT_prolongation: "#4caf50",
      ST_depression: "#e91e63",
      ST_elevation: "#f44336",
      Atrial_Fibrillation: "#9c27b0",
      Heart_block: "#795548",
    };

    return colorMap[type] || "#607d8b"; // Default color if type not found
  };

  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1rem",
        }}
      >
        <h2>Heart Failure Prediction Results</h2>
        <button
          onClick={() => navigate("/history")}
          style={{
            padding: "0.5rem 1rem",
            backgroundColor: "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          View Patient History
        </button>
      </div>

      <div className="results-container">
        <div className="results-card">
          <h3 className="results-title">Risk Assessment</h3>
          <p>Patient: {predictionResult.patient_id}</p>
          <p>
            Risk Level: <span className={riskClass}>{riskLevel}</span>
          </p>
          <div className={`risk-score ${riskClass}`}>
            {(predictionResult.prediction * 100).toFixed(1)}%
          </div>
          <p>Confidence: {(predictionResult.confidence * 100).toFixed(1)}%</p>

          {predictionResult.retraining_info && (
            <div className="retraining-info">
              <h4>Model Information</h4>
              <p>
                Records since last retraining:{" "}
                {predictionResult.retraining_info.records_since_last_retraining}
              </p>
              {predictionResult.retraining_info.drift_detected && (
                <p className="drift-warning">
                  Model drift detected! The model will be retrained.
                </p>
              )}
              <p style={{ marginTop: "1rem" }}>
                <a
                  href="/retrain"
                  className="view-button"
                  style={{ textDecoration: "none", display: "inline-block" }}
                >
                  View Retraining Options
                </a>
              </p>
            </div>
          )}
        </div>

        <div className="results-card">
          <h3 className="results-title">Feature Importance</h3>
          <div style={{ height: "300px" }}>
            <canvas ref={shapChartRef}></canvas>
          </div>
        </div>
      </div>

      {/* Biomarker Impact Chart - Only show if NT-proBNP is present */}
      {predictionResult.patient_data?.biomarkers?.nt_probnp && (
        <div className="results-card" style={{ marginTop: "2rem" }}>
          <h3 className="results-title">NT-proBNP Biomarker Analysis</h3>
          <div style={{ padding: "1rem 0" }}>
            <p style={{ marginBottom: "1rem" }}>
              <strong>Patient NT-proBNP Value:</strong>{" "}
              {predictionResult.patient_data.biomarkers.nt_probnp} pg/mL
            </p>
            <BiomarkerImpactChart
              patientAge={predictionResult.patient_data.age}
            />
          </div>
        </div>
      )}

      <div className="results-card" style={{ marginTop: "2rem" }}>
        <h3 className="results-title">ECG Analysis</h3>
        <div className="ecg-container">
          <canvas ref={ecgChartRef}></canvas>
        </div>

        <h4>Abnormality Timeline</h4>
        <div className="abnormality-timeline">
          {/* Count total abnormalities */}
          {(() => {
            const totalAbnormalities = Object.values(
              predictionResult.abnormalities
            ).flat().length;

            // Log abnormality data for debugging
            console.log("Abnormality data:", predictionResult.abnormalities);
            console.log("Total abnormalities:", totalAbnormalities);

            return (
              <div className="abnormality-summary">
                <p>
                  {totalAbnormalities === 0
                    ? "No abnormalities detected"
                    : `${totalAbnormalities} ${
                        totalAbnormalities === 1
                          ? "abnormality"
                          : "abnormalities"
                      } detected`}
                </p>
              </div>
            );
          })()}

          {/* Display abnormality timeline */}
          <div className="timeline-container">
            {/* Timeline axis */}
            <div className="timeline-axis">
              {[...Array(11)].map((_, i) => (
                <div key={i} className="timeline-tick">
                  <div className="tick-mark"></div>
                  <div className="tick-label">{i}s</div>
                </div>
              ))}
            </div>

            {/* Abnormality markers */}
            <div className="timeline-events">
              {Object.entries(predictionResult.abnormalities).map(
                ([type, instances]) =>
                  instances.length > 0 && (
                    <div key={type} className="abnormality-type-row">
                      <div className="abnormality-type-label">
                        {type.replace(/_/g, " ")}
                      </div>
                      <div className="abnormality-markers">
                        {instances.map((instance, index) => {
                          // Calculate position based on time
                          const time =
                            typeof instance.time === "number"
                              ? instance.time
                              : parseFloat(instance.time) || 0;
                          const duration =
                            typeof instance.duration === "number"
                              ? instance.duration
                              : parseFloat(instance.duration) || 0.5;

                          // Calculate position and width
                          const startPercent = (time / 10) * 100;
                          const widthPercent = (duration / 10) * 100;

                          return (
                            <div
                              key={index}
                              className={`abnormality-marker ${type.toLowerCase()}`}
                              style={{
                                left: `${startPercent}%`,
                                width: `${widthPercent}%`,
                              }}
                              title={
                                instance.description ||
                                `${type.replace(
                                  /_/g,
                                  " "
                                )} at ${instance.time.toFixed(2)}s`
                              }
                            >
                              <div className="marker-tooltip">
                                <strong>{type.replace(/_/g, " ")}</strong>
                                <p>
                                  {instance.description ||
                                    `Detected at ${instance.time.toFixed(2)}s`}
                                </p>
                                {instance.duration && (
                                  <p>
                                    Duration: {instance.duration.toFixed(2)}s
                                  </p>
                                )}
                                {instance.rate && (
                                  <p>Rate: {Math.round(instance.rate)} bpm</p>
                                )}
                                {instance.interval && (
                                  <p>
                                    Interval:{" "}
                                    {(instance.interval * 1000).toFixed(0)} ms
                                  </p>
                                )}
                                {instance.magnitude && (
                                  <p>
                                    Magnitude: {instance.magnitude.toFixed(2)}{" "}
                                    mV
                                  </p>
                                )}
                                {instance.degree && (
                                  <p>Degree: {instance.degree}</p>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )
              )}
            </div>
          </div>

          {/* Detailed abnormality list */}
          <div className="abnormality-details">
            <h5>Detailed Findings</h5>
            {Object.entries(predictionResult.abnormalities).map(
              ([type, instances]) =>
                instances.length > 0 && (
                  <div key={type} className="abnormality-type-details">
                    <h6>
                      {type.replace(/_/g, " ")} ({instances.length})
                    </h6>
                    {instances.map((instance, index) => (
                      <div key={index} className="abnormality-detail-item">
                        <div
                          className="abnormality-detail-marker"
                          style={{ backgroundColor: getAbnormalityColor(type) }}
                        ></div>
                        <div className="abnormality-detail-content">
                          <p className="abnormality-description">
                            {instance.description ||
                              `${type.replace(/_/g, " ")} detected`}
                          </p>
                          <p className="abnormality-time">
                            Time: {instance.time.toFixed(2)}s
                            {instance.duration &&
                              ` (duration: ${instance.duration.toFixed(2)}s)`}
                          </p>
                          {instance.rate && (
                            <p>Heart rate: {Math.round(instance.rate)} bpm</p>
                          )}
                          {instance.interval && (
                            <p>
                              QT interval:{" "}
                              {(instance.interval * 1000).toFixed(0)} ms
                            </p>
                          )}
                          {instance.magnitude && (
                            <p>Magnitude: {instance.magnitude.toFixed(2)} mV</p>
                          )}
                          {instance.degree && <p>Degree: {instance.degree}</p>}
                        </div>
                      </div>
                    ))}
                  </div>
                )
            )}

            {Object.values(predictionResult.abnormalities).every(
              (arr) => arr.length === 0
            ) && <p>No abnormalities detected</p>}
          </div>
        </div>

        {/* Patient verification status */}
        <div style={{ marginTop: "2rem", textAlign: "center" }}>
          <div
            style={{
              padding: "1rem",
              backgroundColor: patientVerified
                ? "#e8f5e9"
                : patientError
                ? "#ffebee"
                : "#fff8e1",
              borderRadius: "8px",
              maxWidth: "600px",
              margin: "0 auto",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            }}
          >
            {patientVerified ? (
              <>
                <p
                  style={{
                    marginBottom: "1rem",
                    fontWeight: "bold",
                    color: "#2e7d32",
                  }}
                >
                  ✅ Patient data has been saved successfully! You can view all
                  patients in the history page.
                </p>
                <button
                  onClick={() => navigate("/history")}
                  style={{
                    padding: "0.75rem 1.5rem",
                    backgroundColor: "#4CAF50",
                    color: "white",
                    border: "none",
                    borderRadius: "4px",
                    cursor: "pointer",
                    fontSize: "1rem",
                    fontWeight: "bold",
                    boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                  }}
                >
                  View Patient History
                </button>
              </>
            ) : patientError ? (
              <>
                <p
                  style={{
                    marginBottom: "1rem",
                    fontWeight: "bold",
                    color: "#c62828",
                  }}
                >
                  ❌ {patientError}
                </p>
                <div
                  style={{
                    display: "flex",
                    gap: "10px",
                    justifyContent: "center",
                  }}
                >
                  <button
                    onClick={() => navigate("/")}
                    style={{
                      padding: "0.75rem 1.5rem",
                      backgroundColor: "#f44336",
                      color: "white",
                      border: "none",
                      borderRadius: "4px",
                      cursor: "pointer",
                      fontSize: "1rem",
                      fontWeight: "bold",
                    }}
                  >
                    Try Again
                  </button>
                  <button
                    onClick={() => navigate("/history")}
                    style={{
                      padding: "0.75rem 1.5rem",
                      backgroundColor: "#9e9e9e",
                      color: "white",
                      border: "none",
                      borderRadius: "4px",
                      cursor: "pointer",
                      fontSize: "1rem",
                      fontWeight: "bold",
                    }}
                  >
                    View History Anyway
                  </button>
                </div>
              </>
            ) : (
              <>
                <p
                  style={{
                    marginBottom: "1rem",
                    fontWeight: "bold",
                    color: "#f57c00",
                  }}
                >
                  ⏳ Verifying patient data... (Attempt{" "}
                  {verificationAttempts + 1}/{MAX_VERIFICATION_ATTEMPTS})
                </p>
                <div
                  className="spinner"
                  style={{
                    margin: "0 auto",
                    width: "30px",
                    height: "30px",
                    border: "4px solid rgba(0, 0, 0, 0.1)",
                    borderRadius: "50%",
                    borderTop: "4px solid #f57c00",
                    animation: "spin 1s linear infinite",
                  }}
                ></div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;
