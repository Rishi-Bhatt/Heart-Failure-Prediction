import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import "../styles/new-patient-history.css";

const NewPatientHistory = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const navigate = useNavigate();

  // Function to fetch patients
  const fetchPatients = async () => {
    setLoading(true);
    try {
      console.log("Fetching patients...");

      // Add timestamp for cache busting
      const timestamp = new Date().getTime();

      const response = await axios.get("http://localhost:8083/api/patients", {
        params: {
          limit: 100,
          t: timestamp,
        },
        headers: {
          "Cache-Control": "no-cache, no-store, must-revalidate",
          Pragma: "no-cache",
          Expires: "0",
        },
      });

      console.log("API Response:", response.data);
      console.log(`Found ${response.data.length} patients`);

      setPatients(response.data);
      setLastUpdated(new Date());
      setLoading(false);
    } catch (err) {
      console.error("Error fetching patients:", err);
      setError("Failed to load patients. Please try again.");
      setLoading(false);
    }
  };

  // Fetch patients on component mount
  useEffect(() => {
    console.log(
      "NewPatientHistory component mounted - fetching patients immediately"
    );

    // Fetch immediately
    fetchPatients();

    // Then fetch again after a short delay to ensure we get the latest data
    const initialTimeout = setTimeout(() => {
      console.log("Fetching patients again after initial delay");
      fetchPatients();
    }, 1000);

    // Set up polling every 5 seconds
    const intervalId = setInterval(() => {
      console.log("Polling for patients");
      fetchPatients();
    }, 5000);

    // Clean up interval and timeout on unmount
    return () => {
      clearInterval(intervalId);
      clearTimeout(initialTimeout);
    };
  }, []);

  // Function to handle refresh button click
  const handleRefresh = () => {
    fetchPatients();
  };

  // Function to create a test patient
  const createTestPatient = async () => {
    try {
      setLoading(true);
      const response = await axios.get(
        "http://localhost:8083/api/patients/create-test"
      );
      console.log("Created test patient:", response.data);
      fetchPatients();
    } catch (err) {
      console.error("Error creating test patient:", err);
      setError("Failed to create test patient. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Function to navigate to patient details
  const viewPatientDetails = (patientId) => {
    navigate(`/patients/${patientId}`);
  };

  // Function to format date
  const formatDate = (dateString) => {
    if (!dateString) return "Unknown";
    try {
      return new Date(dateString).toLocaleString();
    } catch (e) {
      return dateString;
    }
  };

  // Function to get risk level class
  const getRiskLevelClass = (prediction) => {
    if (prediction >= 0.28) return "high-risk";
    if (prediction >= 0.12) return "medium-risk";
    return "low-risk";
  };

  if (loading && patients.length === 0) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Loading patients...</p>
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
    <div className="new-patient-history">
      <div className="header">
        <h2>Patient History</h2>
        <div className="controls">
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
        <div className="no-patients">
          <p>No patients found.</p>
          <Link to="/" className="add-patient-link">
            Add a new patient
          </Link>
        </div>
      ) : (
        <div className="table-container">
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

      <div className="add-patient-container">
        <Link to="/" className="add-patient-button">
          Add New Patient
        </Link>
        <button onClick={createTestPatient} className="test-patient-button">
          Create Test Patient
        </button>
      </div>
    </div>
  );
};

export default NewPatientHistory;
