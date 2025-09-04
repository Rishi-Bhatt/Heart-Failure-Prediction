import React, { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";

// Set up polling interval (in milliseconds)
const POLLING_INTERVAL = 3000; // 3 seconds

const PatientHistoryManager = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const [patientCount, setPatientCount] = useState(0);
  const navigate = useNavigate();

  // Use useCallback to memoize the fetchPatients function
  const fetchPatients = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }

    try {
      // Add a cache-busting parameter to prevent caching
      const timestamp = new Date().getTime();
      console.log(`Fetching patients with timestamp: ${timestamp}`);

      // Force browser to bypass cache completely
      const response = await axios.get(`http://localhost:8083/api/patients`, {
        timeout: 10000, // 10 second timeout
        headers: {
          "Cache-Control": "no-cache, no-store, must-revalidate",
          Pragma: "no-cache",
          Expires: "0",
        },
        // Prevent axios from caching and ensure we get all patients
        params: {
          t: timestamp, // Cache busting parameter
          _: timestamp, // Additional cache busting parameter
          limit: 1000, // Set a high limit to get all patients
        },
      });

      // Log the response headers for debugging
      console.log("Response headers:", response.headers);
      console.log(`X-Patient-Count: ${response.headers["x-patient-count"]}`);
      console.log(`X-Timestamp: ${response.headers["x-timestamp"]}`);

      // Get response headers for debugging
      const patientCount = response.headers["x-patient-count"] || "unknown";
      const responseTimestamp = response.headers["x-timestamp"] || "none";

      console.log("Fetched patients:", response.data);
      console.log(
        `Response headers - Count: ${patientCount}, Timestamp: ${responseTimestamp}`
      );

      // The response is now directly an array
      const patientsArray = response.data;
      console.log(
        `Received ${patientsArray.length} patients directly as array`
      );

      // Process the data to ensure it has the correct structure
      const processedData = patientsArray.map((patient) => {
        // Extract file modification time if available
        const fileInfo = patient.file_mtime
          ? ` (File: ${new Date(patient.file_mtime * 1000).toLocaleString()})`
          : "";

        return {
          ...patient,
          // Ensure these fields exist with default values if missing
          patient_id: patient.patient_id || "unknown",
          name: (patient.name || "Unknown") + fileInfo,
          age: patient.age || 0,
          gender: patient.gender || "Unknown",
          timestamp: patient.timestamp || new Date().toISOString(),
          prediction: patient.prediction || 0,
          confidence: patient.confidence || 0,
        };
      });

      console.log("Processed patient data:", processedData);
      setPatients(processedData);
      setPatientCount(processedData.length);
      setLastUpdated(new Date());

      if (showLoading) {
        setLoading(false);
      }
      return true; // Success
    } catch (err) {
      console.error("Error fetching patient history:", err);
      setError("Failed to load patient history. Please try again later.");
      if (showLoading) {
        setLoading(false);
      }
      return false; // Failed
    }
  }, []);

  // Set up polling to refresh patient data periodically
  useEffect(() => {
    // Initial fetch
    fetchPatients();

    // Set up polling interval
    const intervalId = setInterval(() => {
      console.log("Polling for patient updates...");
      fetchPatients(false); // Don't show loading indicator for polling
    }, POLLING_INTERVAL);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, [fetchPatients]);

  // Function to handle manual refresh
  const handleRefresh = () => {
    console.log("Manual refresh requested");
    fetchPatients(true);
  };

  // Function to navigate to patient details
  const viewPatientDetails = (patientId) => {
    console.log(`Navigating to patient details: /patients/${patientId}`);
    navigate(`/patients/${patientId}`);
  };

  if (loading && patients.length === 0) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading patient history...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h3>Error</h3>
        <p>{error}</p>
        <button onClick={handleRefresh} className="refresh-button">
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="patient-history-container">
      <div className="patient-history-header">
        <h2>Patient History</h2>
        <div className="patient-history-controls">
          <div className="patient-count">
            {patientCount} {patientCount === 1 ? "patient" : "patients"}
          </div>
          <button onClick={handleRefresh} className="refresh-button">
            Refresh
          </button>
          <div className="last-updated">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {patients.length === 0 ? (
        <div className="no-patients-message">
          <p>No patient records found.</p>
          <Link to="/new-patient" className="add-patient-link">
            Add a new patient
          </Link>
        </div>
      ) : (
        <div className="patient-table-container">
          <table className="patient-table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Name</th>
                <th>Age</th>
                <th>Gender</th>
                <th>Date</th>
                <th>Risk Score</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((patient) => (
                <tr key={patient.patient_id}>
                  <td>{patient.patient_id}</td>
                  <td>{patient.name}</td>
                  <td>{patient.age}</td>
                  <td>{patient.gender}</td>
                  <td>{new Date(patient.timestamp).toLocaleString()}</td>
                  <td>
                    <div
                      className={`risk-score ${
                        patient.prediction >= 0.28
                          ? "high-risk"
                          : patient.prediction >= 0.12
                          ? "medium-risk"
                          : "low-risk"
                      }`}
                    >
                      {(patient.prediction * 100).toFixed(1)}%
                    </div>
                  </td>
                  <td>
                    <button
                      className="view-details-button"
                      onClick={() => viewPatientDetails(patient.patient_id)}
                    >
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="add-patient-button-container">
        <Link to="/new-patient" className="add-patient-button">
          Add New Patient
        </Link>
      </div>
    </div>
  );
};

export default PatientHistoryManager;
