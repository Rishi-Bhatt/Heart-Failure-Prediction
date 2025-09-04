import { useState, useEffect, useRef } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import axios from "axios";
import Chart from "chart.js/auto";
import CounterfactualExplanation from "./CounterfactualExplanation";
import TwelveLeadECG from "./TwelveLeadECG";
import ECGAbnormalityDisplay from "./ECGAbnormalityDisplay";

const PatientDetail = () => {
  const { patientId } = useParams();
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showTwelveLeadECG, setShowTwelveLeadECG] = useState(false);
  const navigate = useNavigate();

  const ecgChartRef = useRef(null);
  const shapChartRef = useRef(null);

  useEffect(() => {
    const fetchPatientDetails = async () => {
      try {
        console.log(`Fetching details for patient ID: ${patientId}`);

        // Add cache-busting parameter
        const timestamp = new Date().getTime();

        const response = await axios.get(
          `http://localhost:8083/api/patients/${patientId}?t=${timestamp}`,
          {
            headers: {
              "Cache-Control": "no-cache, no-store, must-revalidate",
              Pragma: "no-cache",
              Expires: "0",
            },
            timeout: 10000, // 10 second timeout
          }
        );

        console.log("Fetched patient details:", response.data);

        if (!response.data) {
          throw new Error("No patient data returned from server");
        }

        // Ensure patient data has all required fields
        const patientData = response.data;

        // Initialize patient_data if missing
        if (!patientData.patient_data) {
          patientData.patient_data = {};
        }

        // Initialize abnormalities if missing
        if (!patientData.abnormalities) {
          patientData.abnormalities = {};
        }

        // Initialize ecg_signal and ecg_time if missing
        if (!patientData.ecg_signal) {
          patientData.ecg_signal = [];
        }

        if (!patientData.ecg_time) {
          patientData.ecg_time = [];
        }

        // Initialize shap_values if missing
        if (!patientData.shap_values) {
          patientData.shap_values = {
            feature_names: [],
            values: [],
            base_value: 0.5,
          };
        }

        setPatient(patientData);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching patient details:", err);
        setError(
          `Failed to load patient details for ID ${patientId}. Please try again later.`
        );
        setLoading(false);
      }
    };

    fetchPatientDetails();
  }, [patientId]);

  // Cleanup function for charts when toggling between views
  useEffect(() => {
    // When toggling to 12-lead ECG, destroy the single-lead chart
    if (showTwelveLeadECG && ecgChartRef.current && ecgChartRef.current.chart) {
      ecgChartRef.current.chart.destroy();
      ecgChartRef.current.chart = null;
    }
  }, [showTwelveLeadECG]);

  // Create ECG chart
  useEffect(() => {
    if (patient && ecgChartRef.current && !showTwelveLeadECG) {
      try {
        console.log("Attempting to create ECG chart");
        const ecgCtx = ecgChartRef.current.getContext("2d");

        // Destroy previous chart if it exists
        if (ecgChartRef.current.chart) {
          ecgChartRef.current.chart.destroy();
        }

        // Check if we have ECG data
        if (
          !patient.ecg_signal ||
          patient.ecg_signal.length === 0 ||
          !patient.ecg_time ||
          patient.ecg_time.length === 0
        ) {
          console.log("No ECG data available");
          // Create a simple placeholder chart
          const ecgChart = new Chart(ecgCtx, {
            type: "line",
            data: {
              labels: [0, 1, 2, 3, 4, 5],
              datasets: [
                {
                  label: "ECG Signal (Sample)",
                  data: [0, 0.5, 0, 0.5, 0, 0.5],
                  borderColor: "rgb(200, 200, 200)",
                  borderWidth: 1,
                  borderDash: [5, 5],
                  pointRadius: 0,
                },
              ],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                title: {
                  display: true,
                  text: "No ECG Data Available",
                },
              },
            },
          });
          ecgChartRef.current.chart = ecgChart;
          return;
        }

        console.log("ECG data available, creating chart");
        console.log(
          `ECG signal length: ${patient.ecg_signal.length}, ECG time length: ${patient.ecg_time.length}`
        );

        // Create abnormality regions
        const abnormalityRegions = [];

        // Add regions for each abnormality type
        if (patient.abnormalities) {
          Object.entries(patient.abnormalities).forEach(([type, instances]) => {
            if (instances && instances.length > 0) {
              instances.forEach((instance) => {
                if (!instance) return;

                const startTime =
                  typeof instance.time === "number"
                    ? instance.time
                    : parseFloat(instance.time) || 0;
                const duration =
                  typeof instance.duration === "number"
                    ? instance.duration
                    : parseFloat(instance.duration) || 0.5;
                const endTime = startTime + duration;

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
          });
        }

        // Create ECG chart
        const ecgChart = new Chart(ecgCtx, {
          type: "line",
          data: {
            labels: patient.ecg_time,
            datasets: [
              {
                label: "ECG Signal",
                data: patient.ecg_signal,
                borderColor: "rgb(75, 192, 192)",
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1,
                cubicInterpolationMode: "monotone",
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
              duration: 1000, // Animate for 1 second when data changes
            },
            scales: {
              x: {
                title: {
                  display: true,
                  text: "Time (seconds)",
                },
              },
              y: {
                title: {
                  display: true,
                  text: "Amplitude",
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
      } catch (error) {
        console.error("Error creating ECG chart:", error);
      }
    }
  }, [patient, showTwelveLeadECG]);

  // Create SHAP chart
  useEffect(() => {
    if (patient && shapChartRef.current) {
      try {
        console.log("Attempting to create SHAP chart");
        const shapCtx = shapChartRef.current.getContext("2d");

        // Destroy previous chart if it exists
        if (shapChartRef.current.chart) {
          shapChartRef.current.chart.destroy();
        }

        // Check if we have SHAP values
        if (
          !patient.shap_values ||
          !patient.shap_values.feature_names ||
          !patient.shap_values.values ||
          patient.shap_values.feature_names.length === 0
        ) {
          console.log("No SHAP values available, creating placeholder chart");

          // Create a placeholder chart with sample data
          const sampleFeatures = [
            { name: "Age", value: 0.05 },
            { name: "Blood Pressure", value: 0.04 },
            { name: "Cholesterol", value: 0.03 },
            { name: "Max Heart Rate", value: -0.02 },
            { name: "ST Depression", value: 0.06 },
          ];

          const backgroundColors = sampleFeatures.map((feature) =>
            feature.value >= 0
              ? "rgba(200, 200, 200, 0.5)"
              : "rgba(150, 150, 150, 0.5)"
          );

          const shapChart = new Chart(shapCtx, {
            type: "bar",
            data: {
              labels: sampleFeatures.map((f) => f.name),
              datasets: [
                {
                  label: "Feature Importance (Sample Data)",
                  data: sampleFeatures.map((f) => f.value),
                  backgroundColor: backgroundColors,
                  borderColor: backgroundColors.map((c) =>
                    c.replace("0.5", "0.8")
                  ),
                  borderWidth: 1,
                  borderDash: [5, 5],
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
                  text: "No Feature Importance Data Available",
                },
              },
            },
          });

          shapChartRef.current.chart = shapChart;
          return;
        }

        console.log("SHAP values available, creating chart");

        // Get top 10 features by absolute SHAP value
        const shapData = patient.shap_values;

        // Handle nested array structure of SHAP values
        let shapValues = shapData.values;
        if (
          Array.isArray(shapValues) &&
          shapValues.length > 0 &&
          Array.isArray(shapValues[0])
        ) {
          console.log("Detected nested SHAP values array, using first element");
          shapValues = shapValues[0];
        }

        // Log SHAP values for debugging
        console.log("SHAP values structure:", {
          type: typeof shapValues,
          isArray: Array.isArray(shapValues),
          length: Array.isArray(shapValues) ? shapValues.length : "N/A",
          sample: Array.isArray(shapValues)
            ? shapValues.slice(0, 3)
            : shapValues,
        });

        // Check if feature_names and values have the same length
        if (shapData.feature_names.length !== shapValues.length) {
          console.warn(
            `SHAP feature_names (${shapData.feature_names.length}) and values (${shapValues.length}) have different lengths. This may cause issues.`
          );
        }

        // Create feature importance objects with name and value
        const featureImportance = shapData.feature_names.map((name, index) => ({
          name: name,
          value:
            Array.isArray(shapValues) && index < shapValues.length
              ? shapValues[index] || 0
              : 0,
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
      } catch (error) {
        console.error("Error creating SHAP chart:", error);
      }
    }
  }, [patient]);

  // Function to go back to patient history
  const goBackToHistory = () => {
    console.log("Navigating back to patient history");
    navigate("/history");
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading patient details...</p>
        <button onClick={goBackToHistory} className="back-button">
          &larr; Back to Patient History
        </button>
      </div>
    );
  }

  if (error) {
    return (
      <div>
        <div className="error-message">{error}</div>
        <button onClick={goBackToHistory} className="back-button">
          &larr; Back to Patient History
        </button>
      </div>
    );
  }

  if (!patient) {
    return (
      <div>
        <div className="error-message">Patient not found</div>
        <button onClick={goBackToHistory} className="back-button">
          &larr; Back to Patient History
        </button>
      </div>
    );
  }

  // Get risk level from backend or calculate as fallback
  const riskLevel =
    patient.risk_category ||
    (patient.prediction >= 0.28
      ? "High"
      : patient.prediction >= 0.12
      ? "Medium"
      : "Low");

  const riskClass = `risk-${riskLevel.toLowerCase()}`;

  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  return (
    <div>
      <div style={{ marginBottom: "1rem" }}>
        <button onClick={goBackToHistory} className="back-button">
          &larr; Back to Patient History
        </button>
      </div>

      <h2>Patient Details</h2>

      <div
        className="patient-navigation"
        style={{
          display: "flex",
          marginBottom: "20px",
          borderBottom: "1px solid #eee",
          paddingBottom: "10px",
        }}
      >
        <span
          className="nav-link active"
          style={{
            padding: "8px 16px",
            marginRight: "10px",
            fontWeight: "bold",
            color: "#3498db",
          }}
        >
          Patient Summary
        </span>
        <Link
          to={`/patients/${patientId}/trajectory`}
          className="nav-link"
          style={{
            padding: "8px 16px",
            marginRight: "10px",
            color: "#666",
            textDecoration: "none",
          }}
        >
          Longitudinal Tracking
        </Link>
        <Link
          to={`/patients/${patientId}/forecast`}
          className="nav-link"
          style={{
            padding: "8px 16px",
            marginRight: "10px",
            color: "#666",
            textDecoration: "none",
          }}
        >
          Risk Forecast
        </Link>
      </div>

      <div className="form-container">
        <h3 className="form-title">
          {patient.patient_data.name || "Unknown"} -{" "}
          {formatDate(patient.timestamp || new Date())}
        </h3>

        <div className="form-section">
          <h4 className="form-section-title">Basic Information</h4>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Age</label>
              <p>{patient.patient_data.age || "N/A"}</p>
            </div>
            <div className="form-group">
              <label className="form-label">Gender</label>
              <p>{patient.patient_data.gender || "N/A"}</p>
            </div>
            <div className="form-group">
              <label className="form-label">Blood Pressure</label>
              <p>{patient.patient_data.blood_pressure || "N/A"}</p>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h4 className="form-section-title">Clinical Measurements</h4>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Cholesterol</label>
              <p>
                {patient.patient_data.cholesterol
                  ? `${patient.patient_data.cholesterol} mg/dL`
                  : "N/A"}
              </p>
            </div>
            <div className="form-group">
              <label className="form-label">Fasting Blood Sugar</label>
              <p>
                {patient.patient_data.fasting_blood_sugar
                  ? `${patient.patient_data.fasting_blood_sugar} mg/dL`
                  : "N/A"}
              </p>
            </div>
            <div className="form-group">
              <label className="form-label">Max Heart Rate</label>
              <p>
                {patient.patient_data.max_heart_rate
                  ? `${patient.patient_data.max_heart_rate} bpm`
                  : "N/A"}
              </p>
            </div>
          </div>
        </div>

        {patient.patient_data.prior_cardiac_event &&
          patient.patient_data.prior_cardiac_event.type && (
            <div className="form-section">
              <h4 className="form-section-title">Prior Cardiac Event</h4>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Type</label>
                  <p>{patient.patient_data.prior_cardiac_event.type}</p>
                </div>
                <div className="form-group">
                  <label className="form-label">Time Since Event</label>
                  <p>
                    {patient.patient_data.prior_cardiac_event.time_since_event}{" "}
                    months
                  </p>
                </div>
                <div className="form-group">
                  <label className="form-label">Severity</label>
                  <p>{patient.patient_data.prior_cardiac_event.severity}</p>
                </div>
              </div>
            </div>
          )}

        {patient.patient_data.biomarkers && (
          <div className="form-section">
            <h4 className="form-section-title">Biomarkers</h4>
            <div className="form-row">
              {patient.patient_data.biomarkers.nt_probnp && (
                <div className="form-group">
                  <label className="form-label">NT-proBNP</label>
                  <p>{patient.patient_data.biomarkers.nt_probnp} pg/mL</p>
                </div>
              )}
              {!patient.patient_data.biomarkers.nt_probnp && (
                <div className="form-group">
                  <p>No biomarker data available</p>
                </div>
              )}
            </div>
          </div>
        )}

        {patient.patient_data.medications &&
          patient.patient_data.medications.length > 0 && (
            <div className="form-section">
              <h4 className="form-section-title">Medications</h4>
              <ul>
                {patient.patient_data.medications.map((med, index) => (
                  <li key={index}>
                    {med.type} - {med.time_of_administration} hours post-event
                  </li>
                ))}
              </ul>
            </div>
          )}
      </div>

      <div className="results-container" style={{ marginTop: "2rem" }}>
        <div className="results-card">
          <h3 className="results-title">Risk Assessment</h3>
          <p>
            Risk Level: <span className={riskClass}>{riskLevel}</span>
          </p>
          <div className={`risk-score ${riskClass}`}>
            {(patient.prediction ? patient.prediction * 100 : 0).toFixed(1)}%
          </div>
          <p>
            Confidence:{" "}
            {(patient.confidence ? patient.confidence * 100 : 0).toFixed(1)}%
          </p>
        </div>

        <div className="results-card">
          <h3 className="results-title">Feature Importance</h3>
          <div style={{ height: "300px" }}>
            <canvas ref={shapChartRef}></canvas>
          </div>
        </div>
      </div>

      <div className="results-card" style={{ marginTop: "2rem" }}>
        <div className="ecg-header">
          <h3 className="results-title">ECG Analysis</h3>
          <div className="ecg-toggle">
            <button
              className={`toggle-btn ${!showTwelveLeadECG ? "active" : ""}`}
              onClick={() => setShowTwelveLeadECG(false)}
            >
              Single Lead
            </button>
            <button
              className={`toggle-btn ${showTwelveLeadECG ? "active" : ""}`}
              onClick={() => setShowTwelveLeadECG(true)}
            >
              12-Lead ECG
            </button>
          </div>
        </div>

        {showTwelveLeadECG ? (
          <TwelveLeadECG patientId={patientId} />
        ) : (
          <>
            <div className="ecg-container">
              <canvas ref={ecgChartRef}></canvas>
            </div>

            <h4>Abnormality Timeline</h4>
            <ECGAbnormalityDisplay
              abnormalities={patient.abnormalities}
              leadFilter={null} // No lead filter for single-lead ECG
            />
          </>
        )}
      </div>

      {/* Counterfactual Explanations */}
      <div style={{ marginTop: "2rem" }}>
        <CounterfactualExplanation patientId={patientId} />
      </div>

      {/* Link to Longitudinal Patient Trajectory */}
      <div
        className="results-card"
        style={{ marginTop: "2rem", padding: "20px", textAlign: "center" }}
      >
        <h3>Longitudinal Patient Tracking</h3>
        <p>
          View the patient's risk trajectory and biomarker changes over time.
        </p>
        <Link
          to={`/patients/${patientId}/trajectory`}
          style={{
            display: "inline-block",
            margin: "10px 0",
            padding: "10px 20px",
            backgroundColor: "#3498db",
            color: "white",
            borderRadius: "4px",
            textDecoration: "none",
            fontWeight: "bold",
          }}
        >
          View Longitudinal Tracking
        </Link>
      </div>
    </div>
  );
};

export default PatientDetail;
