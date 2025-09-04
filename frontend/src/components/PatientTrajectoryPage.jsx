import React, { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import axios from "axios";
import PatientTrajectory from "./PatientTrajectory";

/**
 * PatientTrajectoryPage - A standalone page for the patient trajectory view
 */
const PatientTrajectoryPage = () => {
  const { patientId } = useParams();
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch patient data
  useEffect(() => {
    const fetchPatientData = async () => {
      try {
        setLoading(true);
        const response = await axios.get(
          `http://localhost:8083/api/patients/${patientId}`,
          {
            headers: {
              "Cache-Control": "no-cache, no-store, must-revalidate",
              Pragma: "no-cache",
              Expires: "0",
            },
            timeout: 10000, // 10 second timeout
          }
        );

        if (response.data) {
          setPatient(response.data);
        } else {
          throw new Error("No patient data returned from server");
        }
      } catch (err) {
        console.error("Error fetching patient data:", err);
        setError(
          `Failed to load patient data for ID ${patientId}. Please try again later.`
        );
      } finally {
        setLoading(false);
      }
    };

    if (patientId) {
      fetchPatientData();
    }
  }, [patientId]);

  if (loading) {
    return (
      <div className="patient-trajectory-page">
        <div className="loading-indicator">Loading patient data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="patient-trajectory-page">
        <div className="error-message">{error}</div>
        <Link to="/history" className="back-link">
          Return to Patient History
        </Link>
      </div>
    );
  }

  if (!patient) {
    return (
      <div className="patient-trajectory-page">
        <div className="error-message">Patient not found</div>
        <Link to="/history" className="back-link">
          Return to Patient History
        </Link>
      </div>
    );
  }

  return (
    <div className="patient-trajectory-page">
      <div className="page-navigation" style={{
        display: "flex",
        marginBottom: "20px",
        borderBottom: "1px solid #eee",
        paddingBottom: "10px",
      }}>
        <Link
          to={`/patients/${patientId}`}
          className="nav-link"
          style={{
            padding: "8px 16px",
            marginRight: "10px",
            color: "#666",
            textDecoration: "none",
          }}
        >
          Patient Summary
        </Link>
        <span
          className="nav-link active"
          style={{
            padding: "8px 16px",
            marginRight: "10px",
            fontWeight: "bold",
            color: "#3498db",
          }}
        >
          Longitudinal Tracking
        </span>
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

      <PatientTrajectory 
        patientId={patientId} 
        patientData={patient.patient_data || {}} 
        standalone={true}
      />
    </div>
  );
};

export default PatientTrajectoryPage;
