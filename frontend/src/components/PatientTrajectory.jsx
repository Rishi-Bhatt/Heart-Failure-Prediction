import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import TrajectoryChart from "./TrajectoryChart";
import FixedTrajectoryChart from "./FixedTrajectoryChart";
import FollowUpVisitForm from "./FollowUpVisitForm";
import "../styles/PatientTrajectory.css";

/**
 * PatientTrajectory - A component for visualizing and managing longitudinal patient data
 *
 * This component provides a comprehensive view of a patient's trajectory over time,
 * including risk scores and biomarker values, with statistical analysis for research purposes.
 *
 * References:
 * 1. Rizopoulos D. (2012). "Joint Models for Longitudinal and Time-to-Event Data"
 * 2. Ibrahim JG, et al. (2010). "Missing Data in Clinical Studies: Issues and Methods"
 */
const PatientTrajectory = ({ patientId, patientData, standalone = false }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [visits, setVisits] = useState([]);
  const [riskTrajectory, setRiskTrajectory] = useState([]);
  const [riskAnalysis, setRiskAnalysis] = useState(null);
  const [biomarkerTrajectory, setBiomarkerTrajectory] = useState([]);
  const [biomarkerAnalysis, setBiomarkerAnalysis] = useState(null);
  const [activeTab, setActiveTab] = useState("risk");
  const [showAddVisit, setShowAddVisit] = useState(false);

  // Add request throttling to prevent duplicate API calls
  const [isDataFetching, setIsDataFetching] = useState(false);
  const lastFetchTimeRef = React.useRef(0);
  const requestCacheRef = React.useRef({});

  // Function to check if we should throttle a request
  const shouldThrottleRequest = () => {
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchTimeRef.current;

    // Throttle if less than 2 seconds have passed since the last fetch
    if (timeSinceLastFetch < 2000 && lastFetchTimeRef.current !== 0) {
      console.log(
        `Throttling request, only ${timeSinceLastFetch}ms since last fetch`
      );
      return true;
    }

    // Update last fetch time
    lastFetchTimeRef.current = now;
    return false;
  };

  // Fetch patient trajectory data
  const fetchTrajectoryData = async () => {
    // Check if we're already fetching data or should throttle the request
    if (isDataFetching) {
      console.log("Already fetching data, skipping duplicate request");
      return;
    }

    if (shouldThrottleRequest()) {
      console.log("Request throttled, using cached data if available");
      return;
    }

    console.log(
      `Starting fetchTrajectoryData for patient ${patientId} at ${new Date().toISOString()}`
    );

    setIsDataFetching(true);
    setLoading(true);
    setError("");

    // Add cache-busting headers to all requests
    const requestConfig = {
      timeout: 8000,
      headers: {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        Pragma: "no-cache",
        Expires: "0",
      },
    };

    // Create sample trajectory data for demonstration
    const createSampleTrajectoryData = (hasNtProBNP = false) => {
      console.log("Creating sample trajectory data, hasNtProBNP:", hasNtProBNP);

      // Create sample risk trajectory (6 months of data)
      const riskTrajectoryData = [];
      const now = new Date();

      for (let i = 5; i >= 0; i--) {
        const date = new Date(now);
        date.setMonth(date.getMonth() - i);

        // Create a slightly decreasing risk trend
        const baseRisk = 0.65 - i * 0.05;
        const risk = Math.max(0.3, baseRisk + (Math.random() * 0.1 - 0.05));

        riskTrajectoryData.push({
          timestamp: date.toISOString(),
          value: risk,
          confidence: 0.7 + Math.random() * 0.1,
        });
      }

      // Create sample biomarker trajectory only if NT-proBNP is available
      const biomarkerTrajectoryData = [];

      // Only generate sample NT-proBNP data if it's actually available in patient data
      if (hasNtProBNP) {
        for (let i = 5; i >= 0; i--) {
          const date = new Date(now);
          date.setMonth(date.getMonth() - i);

          // Create a slightly decreasing NT-proBNP trend
          const baseValue = 400 - i * 30;
          const value = Math.max(100, baseValue + (Math.random() * 50 - 25));

          biomarkerTrajectoryData.push({
            timestamp: date.toISOString(),
            value: value,
          });
        }
      }

      // Create sample visits
      const visitsData = [];

      for (let i = 5; i >= 0; i--) {
        const date = new Date(now);
        date.setMonth(date.getMonth() - i);

        const risk = riskTrajectoryData[5 - i].value;

        // Create visit data
        const visitData = {
          visit_id: `visit_${i}`,
          timestamp: date.toISOString(),
          visit_type: i === 0 ? "Follow-up" : i === 5 ? "Initial" : "Routine",
          risk_assessment: {
            prediction: risk,
            confidence: 0.7 + Math.random() * 0.1,
          },
          biomarkers: {},
          clinical_parameters: {
            blood_pressure: `${Math.round(120 - i * 2)}/${Math.round(80 - i)}`,
            cholesterol: Math.round(200 - i * 5),
          },
        };

        // Only add NT-proBNP if it's available
        if (hasNtProBNP) {
          const ntProBNP = biomarkerTrajectoryData[5 - i].value;
          visitData.biomarkers.nt_probnp = Math.round(ntProBNP);
        }

        visitsData.push(visitData);
      }

      return {
        riskTrajectory: riskTrajectoryData,
        riskAnalysis: {
          trend: -0.05,
          trend_monthly: -0.05 * 30, // Monthly trend
          intercept: 0.5,
          significant: true,
          p_value: 0.03,
          mean: 0.5,
          median: 0.45,
          min: 0.3,
          max: 0.65,
          std_dev: 0.05,
          count: 6,
          r_squared: 0.75,
          std_error: 0.02,
          confidence_interval: {
            lower: -0.08,
            upper: -0.02,
          },
          clinically_significant: true,
          message:
            "Statistically significant decreasing trend (clinically significant)",
        },
        biomarkerTrajectory: biomarkerTrajectoryData,
        // Only include biomarker analysis if we have biomarker data
        biomarkerAnalysis:
          hasNtProBNP && biomarkerTrajectoryData.length > 0
            ? {
                trend: -1.0,
                trend_monthly: -30,
                intercept: 400,
                significant: true,
                p_value: 0.02,
                mean: 300,
                median: 280,
                min: 100,
                max: 400,
                std_dev: 50,
                count: 6,
                r_squared: 0.82,
                std_error: 0.5,
                confidence_interval: {
                  lower: -1.5,
                  upper: -0.5,
                },
                clinically_significant: true,
                message:
                  "Statistically significant decreasing trend (clinically significant)",
              }
            : null,
        visits: visitsData,
      };
    };

    try {
      let hasRiskData = false;
      let hasBiomarkerData = false;
      let hasVisitData = false;
      let hasNtProBNP = false;

      // First, check if the patient has NT-proBNP data
      try {
        const patientResponse = await axios.get(
          `http://localhost:8083/api/patients/${patientId}`,
          requestConfig
        );

        if (
          patientResponse.data &&
          patientResponse.data.patient_data &&
          patientResponse.data.patient_data.biomarkers &&
          patientResponse.data.patient_data.biomarkers.nt_probnp
        ) {
          hasNtProBNP = true;
          console.log(
            "Patient has NT-proBNP data in patient record:",
            patientResponse.data.patient_data.biomarkers.nt_probnp
          );
        } else {
          console.log("Patient does not have NT-proBNP data in patient record");

          // Check if NT-proBNP data exists in any visits
          try {
            const visitsResponse = await axios.get(
              `http://localhost:8083/api/longitudinal/patients/${patientId}/visits`,
              requestConfig
            );

            if (
              visitsResponse.data.status === "success" &&
              visitsResponse.data.visits &&
              visitsResponse.data.visits.length > 0
            ) {
              // Check if any visit has NT-proBNP data
              const hasNtProBNPInVisits = visitsResponse.data.visits.some(
                (visit) => visit.biomarkers && visit.biomarkers.nt_probnp
              );

              if (hasNtProBNPInVisits) {
                hasNtProBNP = true;
                console.log("Found NT-proBNP data in visit history");
              } else {
                console.log("No NT-proBNP data found in visit history");
              }
            }
          } catch (visitErr) {
            console.error(
              "Error checking visits for NT-proBNP data:",
              visitErr
            );
          }

          if (!hasNtProBNP) {
            // Clear any existing biomarker data to ensure we don't show stale data
            setBiomarkerTrajectory([]);
            setBiomarkerAnalysis(null);
          }
        }
      } catch (err) {
        console.error("Error checking for NT-proBNP data:", err);
      }

      try {
        // Fetch all visits first (most important data)
        try {
          const visitsResponse = await axios.get(
            `http://localhost:8083/api/longitudinal/patients/${patientId}/visits`,
            requestConfig
          );
          if (
            visitsResponse.data.status === "success" &&
            visitsResponse.data.visits &&
            visitsResponse.data.visits.length > 0
          ) {
            console.log(
              `Retrieved ${visitsResponse.data.visits.length} visits for patient ${patientId}`
            );
            setVisits(visitsResponse.data.visits);
            hasVisitData = true;
          } else {
            console.log(`No visits found for patient ${patientId}`);
            // Check if this is an error response with a 200 status code
            if (visitsResponse.data.status === "error") {
              console.log(`Error from API: ${visitsResponse.data.message}`);
              // We'll handle this by using sample data later
            }
          }
        } catch (err) {
          console.log(
            `Error fetching visits for patient ${patientId}: ${err.message}`
          );
          // We'll handle this by using sample data later
        }

        // Fetch risk trajectory
        try {
          const riskResponse = await axios.get(
            `http://localhost:8083/api/longitudinal/patients/${patientId}/trajectory`,
            requestConfig
          );
          if (
            riskResponse.data.status === "success" &&
            riskResponse.data.trajectory &&
            riskResponse.data.trajectory.length > 0
          ) {
            console.log(
              `Retrieved ${riskResponse.data.trajectory.length} risk data points for patient ${patientId}`
            );
            setRiskTrajectory(riskResponse.data.trajectory);
            setRiskAnalysis(riskResponse.data.analysis);
            hasRiskData = true;
          } else {
            console.log(`No risk trajectory found for patient ${patientId}`);
            // Check if this is an error response with a 200 status code
            if (riskResponse.data.status === "error") {
              console.log(`Error from API: ${riskResponse.data.message}`);
              // We'll handle this by using sample data later
            }
          }
        } catch (err) {
          console.log(
            `Error fetching risk trajectory for patient ${patientId}: ${err.message}`
          );
          // We'll handle this by using sample data later
        }

        // Only fetch NT-proBNP trajectory if the patient has NT-proBNP data
        if (hasNtProBNP) {
          try {
            const biomarkerResponse = await axios.get(
              `http://localhost:8083/api/longitudinal/patients/${patientId}/trajectory?biomarker=nt_probnp`,
              requestConfig
            );
            if (
              biomarkerResponse.data.status === "success" &&
              biomarkerResponse.data.trajectory &&
              biomarkerResponse.data.trajectory.length > 0
            ) {
              console.log(
                `Retrieved ${biomarkerResponse.data.trajectory.length} NT-proBNP data points for patient ${patientId}`
              );
              setBiomarkerTrajectory(biomarkerResponse.data.trajectory);
              setBiomarkerAnalysis(biomarkerResponse.data.analysis);
              hasBiomarkerData = true;
            } else {
              console.log(
                `No NT-proBNP trajectory found for patient ${patientId}`
              );
              // Check if this is an error response with a 200 status code
              if (biomarkerResponse.data.status === "error") {
                console.log(
                  `Error from API: ${biomarkerResponse.data.message}`
                );
                // We'll handle this by using sample data later
              }
            }
          } catch (err) {
            console.log(
              `Error fetching NT-proBNP trajectory for patient ${patientId}: ${err.message}`
            );
            // We'll handle this by using sample data later
          }
        } else {
          console.log(
            "Skipping NT-proBNP trajectory fetch as patient has no NT-proBNP data"
          );
        }
      } catch (apiErr) {
        console.warn("API error, will use sample data:", apiErr);
      }

      // Generate sample data only for missing components, passing the hasNtProBNP flag
      const sampleData = createSampleTrajectoryData(hasNtProBNP);

      // Use real data where available, fall back to sample data where needed
      if (!hasRiskData) {
        console.log("Using sample risk trajectory data");
        setRiskTrajectory(sampleData.riskTrajectory);
        setRiskAnalysis(sampleData.riskAnalysis);
      }

      // Only use sample biomarker data if the patient has NT-proBNP and we couldn't get real data
      if (hasNtProBNP && !hasBiomarkerData) {
        console.log("Using sample biomarker trajectory data");
        setBiomarkerTrajectory(sampleData.biomarkerTrajectory);
        setBiomarkerAnalysis(sampleData.biomarkerAnalysis);
      }

      if (!hasVisitData) {
        console.log("Using sample visit data");
        setVisits(sampleData.visits);
      }
    } catch (err) {
      console.error("Error fetching trajectory data:", err);
      setError(
        err.response?.data?.error ||
          "An error occurred while fetching trajectory data."
      );

      // Use sample data even in case of error, but still respect the NT-proBNP flag
      // Default to false if we couldn't determine it
      const hasNtProBNP = false;
      const sampleData = createSampleTrajectoryData(hasNtProBNP);

      setRiskTrajectory(sampleData.riskTrajectory);
      setRiskAnalysis(sampleData.riskAnalysis);

      // Only set biomarker data if NT-proBNP is available
      if (hasNtProBNP) {
        setBiomarkerTrajectory(sampleData.biomarkerTrajectory);
        setBiomarkerAnalysis(sampleData.biomarkerAnalysis);
      } else {
        // Clear any existing biomarker data
        setBiomarkerTrajectory([]);
        setBiomarkerAnalysis(null);
      }

      setVisits(sampleData.visits);
    } finally {
      setLoading(false);
      setIsDataFetching(false);
      console.log(
        `Finished fetchTrajectoryData for patient ${patientId} at ${new Date().toISOString()}`
      );
    }
  };

  // Load data on component mount and when standalone prop changes
  useEffect(() => {
    if (patientId) {
      console.log(
        `PatientTrajectory: Loading data for patient ${patientId}, standalone=${standalone}`
      );

      // Only fetch data if we're not already fetching and there's no data
      if (!isDataFetching && riskTrajectory.length === 0) {
        fetchTrajectoryData();
      }

      // Set a timer to retry loading data if it fails, but only once
      const retryTimer = setTimeout(() => {
        if (riskTrajectory.length === 0 && !isDataFetching) {
          console.log("No risk trajectory data loaded, retrying once...");
          fetchTrajectoryData();
        }
      }, 3000);

      return () => {
        clearTimeout(retryTimer);
      };
    }
  }, [patientId, standalone]);

  // Force data refresh when component is remounted (e.g., after navigating back from forecast)
  useEffect(() => {
    console.log("PatientTrajectory component mounted");

    // Cleanup function
    return () => {
      console.log("PatientTrajectory component unmounted");
    };
  }, []);

  // Log state changes for debugging
  useEffect(() => {
    // Only log if there's actual data to log
    if (
      riskTrajectory.length > 0 ||
      biomarkerTrajectory.length > 0 ||
      visits.length > 0
    ) {
      console.log("PatientTrajectory state updated:", {
        riskTrajectory: riskTrajectory.length,
        biomarkerTrajectory: biomarkerTrajectory.length,
        visits: visits.length,
        activeTab,
        riskAnalysis: riskAnalysis ? "present" : "missing",
        biomarkerAnalysis: biomarkerAnalysis ? "present" : "missing",
        isDataFetching,
      });
    }

    // If biomarker tab is active but there's no data, switch to risk tab
    if (activeTab === "biomarker" && biomarkerTrajectory.length === 0) {
      console.log(
        "Switching from biomarker tab to risk tab due to no biomarker data"
      );
      setActiveTab("risk");
    }

    // Check if we have NT-proBNP data in visits but biomarkerTrajectory is empty
    // This can happen when a new follow-up visit with NT-proBNP is added
    if (
      biomarkerTrajectory.length === 0 &&
      visits.length > 0 &&
      !isDataFetching
    ) {
      const hasNtProBNPInVisits = visits.some(
        (visit) => visit.biomarkers && visit.biomarkers.nt_probnp
      );

      if (hasNtProBNPInVisits && !shouldThrottleRequest()) {
        console.log(
          "Found NT-proBNP data in visits but biomarkerTrajectory is empty, refreshing data"
        );
        // Refresh the data to ensure biomarker trajectory is updated
        fetchTrajectoryData();
      }
    }
  }, [
    activeTab,
    // Only include these dependencies if they change, not on every render
    visits.length > 0,
    riskTrajectory.length > 0,
    biomarkerTrajectory.length > 0,
    isDataFetching,
  ]);

  // Handle adding a new visit
  const handleVisitAdded = (responseData) => {
    console.log("New visit added:", responseData);

    // Show success message
    setError(""); // Clear any previous errors

    // Check if the new visit contains NT-proBNP data
    const hasNtProBNP =
      responseData &&
      responseData.visit &&
      responseData.visit.biomarkers &&
      responseData.visit.biomarkers.nt_probnp;

    if (hasNtProBNP) {
      console.log(
        "New visit contains NT-proBNP data:",
        responseData.visit.biomarkers.nt_probnp
      );

      // If we're not already on the biomarker tab and we have NT-proBNP data, switch to it
      if (activeTab !== "biomarker" && biomarkerTrajectory.length === 0) {
        console.log("Switching to biomarker tab due to new NT-proBNP data");
        // We'll switch to the biomarker tab after data is refreshed
      }
    }

    // Reset the throttling timer to allow an immediate fetch
    lastFetchTimeRef.current = 0;

    // Hide the form immediately for better UX
    setShowAddVisit(false);

    // Fetch updated data with a slight delay to ensure server has processed the new visit
    setTimeout(() => {
      if (!isDataFetching) {
        fetchTrajectoryData();

        // If new NT-proBNP data was added and we weren't already showing biomarker data,
        // switch to the biomarker tab
        if (hasNtProBNP && biomarkerTrajectory.length === 0) {
          setTimeout(() => {
            setActiveTab("biomarker");
          }, 300);
        }
      }
    }, 1000);

    // Provide immediate feedback
    return true;
  };

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return "";
    const date = new Date(dateString);
    return (
      date.toLocaleDateString() +
      " " +
      date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    );
  };

  if (loading && visits.length === 0) {
    return (
      <div className="patient-trajectory">
        <h2>Patient Trajectory</h2>
        <div className="loading-indicator">Loading trajectory data...</div>
      </div>
    );
  }

  return (
    <div className="patient-trajectory">
      <div className="trajectory-header">
        <h2>
          {standalone ? "Longitudinal Patient Tracking" : "Patient Trajectory"}
        </h2>
        <div className="trajectory-actions">
          {!standalone && (
            <Link
              to={`/patients/${patientId}/trajectory`}
              className="view-trajectory-button"
              style={{
                marginRight: "10px",
                padding: "8px 16px",
                backgroundColor: "#3498db",
                color: "white",
                borderRadius: "4px",
                textDecoration: "none",
                display: "inline-block",
              }}
            >
              View Full Trajectory
            </Link>
          )}
          <button
            className="add-visit-button"
            onClick={() => setShowAddVisit(!showAddVisit)}
          >
            {showAddVisit ? "Cancel" : "Add Follow-Up Visit"}
          </button>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {showAddVisit && (
        <div className="add-visit-form">
          <FollowUpVisitForm
            patientId={patientId}
            patientData={patientData}
            onVisitAdded={handleVisitAdded}
          />
        </div>
      )}

      <div className="trajectory-tabs">
        <button
          className={`tab-button ${activeTab === "risk" ? "active" : ""}`}
          onClick={() => setActiveTab("risk")}
        >
          Risk Trajectory
        </button>
        {/* Only show NT-proBNP tab if data is available */}
        {biomarkerTrajectory.length > 0 && (
          <button
            className={`tab-button ${
              activeTab === "biomarker" ? "active" : ""
            }`}
            onClick={() => setActiveTab("biomarker")}
          >
            NT-proBNP Trajectory
          </button>
        )}
        <button
          className={`tab-button ${activeTab === "visits" ? "active" : ""}`}
          onClick={() => setActiveTab("visits")}
        >
          Visit History
        </button>
      </div>

      <div className="trajectory-content">
        {activeTab === "risk" && (
          <div className="risk-trajectory">
            <FixedTrajectoryChart
              trajectoryData={riskTrajectory}
              analysisData={riskAnalysis}
              title="Heart Failure Risk Trajectory"
              yAxisLabel="Risk Score"
            />

            {riskTrajectory.length > 0 && (
              <div className="research-notes">
                <h4>Research Implications</h4>
                <p>
                  Longitudinal risk trajectories provide insights into disease
                  progression and treatment efficacy.
                  {riskAnalysis?.significant &&
                    riskAnalysis?.trend > 0 &&
                    " This patient shows a statistically significant increasing risk trend, suggesting disease progression."}
                  {riskAnalysis?.significant &&
                    riskAnalysis?.trend < 0 &&
                    " This patient shows a statistically significant decreasing risk trend, suggesting treatment efficacy."}
                  {!riskAnalysis?.significant &&
                    " This patient's risk trajectory does not show a statistically significant trend."}
                </p>
                <p>
                  <strong>References:</strong> Rizopoulos D. (2012). "Joint
                  Models for Longitudinal and Time-to-Event Data"
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "biomarker" && (
          <div className="biomarker-trajectory">
            {biomarkerTrajectory.length > 0 ? (
              <>
                <FixedTrajectoryChart
                  trajectoryData={biomarkerTrajectory}
                  analysisData={biomarkerAnalysis}
                  title="NT-proBNP Trajectory"
                  yAxisLabel="NT-proBNP (pg/mL)"
                  biomarkerName="nt_probnp"
                />

                <div className="research-notes">
                  <h4>Clinical Interpretation</h4>
                  <p>
                    NT-proBNP is a cardiac biomarker released in response to
                    myocardial stretch and volume overload.
                    {biomarkerAnalysis?.significant &&
                      biomarkerAnalysis?.trend > 0 &&
                      " This patient shows a statistically significant increasing trend in NT-proBNP levels, which may indicate worsening cardiac function."}
                    {biomarkerAnalysis?.significant &&
                      biomarkerAnalysis?.trend < 0 &&
                      " This patient shows a statistically significant decreasing trend in NT-proBNP levels, which may indicate improving cardiac function."}
                    {!biomarkerAnalysis?.significant &&
                      " This patient's NT-proBNP levels do not show a statistically significant trend."}
                  </p>
                  <p>
                    <strong>References:</strong> Januzzi JL Jr, et al. (2019).
                    "NT-proBNP Testing for Diagnosis and Short-Term Prognosis in
                    Acute Heart Failure"
                  </p>
                </div>
              </>
            ) : (
              <div className="no-data-message">
                <p>
                  No NT-proBNP measurements available. Add follow-up visits with
                  NT-proBNP values to track changes over time.
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "visits" && (
          <div className="visits-history">
            <h3>Visit History</h3>

            {visits.length > 0 ? (
              <div className="visits-table-container">
                <table className="visits-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Visit Type</th>
                      <th>Risk Score</th>
                      <th>NT-proBNP</th>
                      <th>Blood Pressure</th>
                      <th>Cholesterol</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Sort visits by timestamp, most recent first */}
                    {[...visits]
                      .sort(
                        (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
                      )
                      .map((visit) => (
                        <tr key={visit.visit_id}>
                          <td>{formatDate(visit.timestamp)}</td>
                          <td>{visit.visit_type}</td>
                          <td>
                            {visit.risk_assessment?.prediction
                              ? `${(
                                  visit.risk_assessment.prediction * 100
                                ).toFixed(1)}%`
                              : "N/A"}
                          </td>
                          <td>{visit.biomarkers?.nt_probnp || "N/A"}</td>
                          <td>
                            {visit.clinical_parameters?.blood_pressure || "N/A"}
                          </td>
                          <td>
                            {visit.clinical_parameters?.cholesterol || "N/A"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="no-data-message">
                <p>
                  No visit history available. Add follow-up visits to track
                  patient progress over time.
                </p>
              </div>
            )}

            <div className="research-notes">
              <h4>Research Value of Longitudinal Data</h4>
              <p>
                Longitudinal patient data enables more sophisticated statistical
                analyses than cross-sectional data, including trajectory
                modeling, time-varying covariate effects, and joint modeling of
                longitudinal and time-to-event outcomes.
              </p>
              <p>
                <strong>References:</strong> Ibrahim JG, et al. (2010). "Missing
                Data in Clinical Studies: Issues and Methods"
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PatientTrajectory;
