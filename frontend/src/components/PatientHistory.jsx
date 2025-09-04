import { useState, useEffect, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";

// Set up polling interval (in milliseconds)
const POLLING_INTERVAL = 5000; // 5 seconds

const PatientHistory = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());
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

      // Log the response for debugging
      console.log("Response status:", response.status);
      console.log("Response data type:", typeof response.data);
      console.log(
        "Response data length:",
        Array.isArray(response.data) ? response.data.length : "not an array"
      );

      // Ensure we have an array of patients
      let patientsArray = [];

      if (Array.isArray(response.data)) {
        patientsArray = response.data;
        console.log(`Received ${patientsArray.length} patients as array`);
      } else if (typeof response.data === "object") {
        // Handle case where response might be an object with patients inside
        if (response.data.patients && Array.isArray(response.data.patients)) {
          patientsArray = response.data.patients;
          console.log(
            `Extracted ${patientsArray.length} patients from response object`
          );
        } else {
          // If it's just an object, convert it to array if it looks like a patient
          if (response.data.patient_id) {
            patientsArray = [response.data];
            console.log("Converted single patient object to array");
          } else {
            console.error(
              "Response data is not in expected format:",
              response.data
            );
          }
        }
      } else {
        console.error("Unexpected response data format:", response.data);
      }

      // Process the data to ensure it has the correct structure
      const processedData = patientsArray
        .filter((patient) => {
          // Filter out any null or undefined entries
          if (!patient) {
            console.warn("Filtered out null/undefined patient entry");
            return false;
          }
          // Filter out entries without patient_id
          if (!patient.patient_id) {
            console.warn("Filtered out patient entry without ID:", patient);
            return false;
          }
          return true;
        })
        .map((patient) => {
          // Extract file modification time if available
          const fileInfo = patient.file_mtime
            ? ` (File: ${new Date(patient.file_mtime * 1000).toLocaleString()})`
            : "";

          // Extract patient name from patient_data if available
          let name = patient.name || "Unknown";
          if (
            !patient.name &&
            patient.patient_data &&
            patient.patient_data.name
          ) {
            name = patient.patient_data.name;
          }

          // Extract age from patient_data if available
          let age = patient.age || 0;
          if (
            (!patient.age || patient.age === 0) &&
            patient.patient_data &&
            patient.patient_data.age
          ) {
            age = patient.patient_data.age;
          }

          // Extract gender from patient_data if available
          let gender = patient.gender || "Unknown";
          if (
            !patient.gender &&
            patient.patient_data &&
            patient.patient_data.gender
          ) {
            gender = patient.patient_data.gender;
          }

          return {
            ...patient,
            // Ensure these fields exist with default values if missing
            patient_id: patient.patient_id || "unknown",
            name: name + fileInfo,
            age: age,
            gender: gender,
            timestamp: patient.timestamp || new Date().toISOString(),
            prediction: patient.prediction || 0,
            confidence: patient.confidence || 0,
          };
        });

      console.log("Processed patient data:", processedData);
      setPatients(processedData);
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
  }, [fetchPatients]); // Include fetchPatients in dependency array

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const getRiskLevelClass = (prediction) => {
    if (prediction >= 0.28) return "risk-high";
    if (prediction >= 0.12) return "risk-medium";
    return "risk-low";
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
        <button onClick={() => fetchPatients(true)} className="refresh-button">
          Try Again
        </button>
      </div>
    );
  }

  // Function to handle manual refresh
  const handleRefresh = () => {
    console.log("Manual refresh requested");
    fetchPatients(true);
  };

  return (
    <div className="patient-history-container">
      <div className="patient-history-header">
        <h2>Patient History</h2>
        <div className="patient-history-controls">
          <div className="patient-count">
            {patients.length} {patients.length === 1 ? "patient" : "patients"}
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
                  <td>{formatDate(patient.timestamp)}</td>
                  <td>
                    <div
                      className={`risk-score ${getRiskLevelClass(
                        patient.prediction
                      )}`}
                    >
                      {(patient.prediction * 100).toFixed(1)}%
                    </div>
                  </td>
                  <td>
                    <button
                      className="view-details-button"
                      onClick={() =>
                        navigate(`/patients/${patient.patient_id}`)
                      }
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
        <Link to="/" className="add-patient-button">
          Add New Patient
        </Link>
      </div>
    </div>
  );
};

export default PatientHistory;
