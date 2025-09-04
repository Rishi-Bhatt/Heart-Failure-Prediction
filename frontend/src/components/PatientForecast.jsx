/**
 * PatientForecast - A component for displaying risk forecasts and scenario modeling
 *
 * This component provides a comprehensive view of a patient's predicted risk trajectory,
 * including scenario modeling for intervention planning.
 *
 * References:
 * 1. Cheng, L., et al. (2020). "Temporal Patterns Mining in Electronic Health Records using Deep Learning"
 * 2. Verma, S., et al. (2023). "Counterfactual Explanations for Machine Learning: A Review of Methods and Applications in Healthcare"
 */
import React, { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";
import axios from "axios";
import ForecastTrajectoryChart from "./ForecastTrajectoryChart";
import FixedForecastChart from "./FixedForecastChart";
import ScenarioModelingTool from "./ScenarioModelingTool";
import "../styles/PatientForecast.css";

const PatientForecast = () => {
  const { patientId } = useParams();
  const [patient, setPatient] = useState(null);
  const [baselineForecast, setBaselineForecast] = useState(null);
  const [scenarios, setScenarios] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch patient data
  useEffect(() => {
    const fetchPatientData = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await axios.get(
          `http://localhost:8083/api/patients/${patientId}`
        );
        setPatient(response.data);
      } catch (err) {
        console.error("Error fetching patient data:", err);
        setError(
          err.response?.data?.error ||
            "An error occurred while fetching patient data"
        );
      } finally {
        setIsLoading(false);
      }
    };

    if (patientId) {
      fetchPatientData();
    }
  }, [patientId]);

  // Handle forecast loaded
  const handleForecastLoaded = (forecastData) => {
    setBaselineForecast(forecastData);
  };

  // Handle scenario update
  const handleScenarioUpdate = (updatedScenarios) => {
    setScenarios(updatedScenarios);
  };

  // If loading, show loading indicator
  if (isLoading) {
    return (
      <div className="patient-forecast">
        <h2>Risk Trajectory Forecast</h2>
        <div className="loading-indicator">Loading patient data...</div>
      </div>
    );
  }

  // If error, show error message
  if (error) {
    return (
      <div className="patient-forecast">
        <h2>Risk Trajectory Forecast</h2>
        <div className="error-message">
          <p>Error: {error}</p>
          <p>
            <Link to="/patients">Return to Patients List</Link>
          </p>
        </div>
      </div>
    );
  }

  // If no patient data, show message
  if (!patient) {
    return (
      <div className="patient-forecast">
        <h2>Risk Trajectory Forecast</h2>
        <div className="no-data-message">
          <p>Patient not found. Please select a valid patient.</p>
          <p>
            <Link to="/patients">Return to Patients List</Link>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="patient-forecast">
      <div className="forecast-header">
        <h2>Risk Trajectory Forecast</h2>
        <div className="patient-info">
          <h3>{patient.patient_data.name}</h3>
          <p>
            {patient.patient_data.age} years, {patient.patient_data.gender}
          </p>
        </div>
      </div>

      <div className="forecast-navigation">
        <Link to={`/patients/${patientId}`} className="nav-link">
          Patient Summary
        </Link>
        <Link to={`/patients/${patientId}/trajectory`} className="nav-link">
          Longitudinal Tracking
        </Link>
        <span className="nav-link active">Risk Forecast</span>
      </div>

      <div className="forecast-content">
        <div className="forecast-chart-section">
          <FixedForecastChart
            patientId={patientId}
            onForecastLoaded={handleForecastLoaded}
          />
        </div>

        <div className="scenario-modeling-section">
          <ScenarioModelingTool
            patientId={patientId}
            baselineForecast={baselineForecast}
            onScenarioUpdate={handleScenarioUpdate}
          />
        </div>
      </div>

      <div className="research-notes">
        <h4>Research Notes</h4>
        <p>
          Longitudinal risk forecasting combines historical patient data with
          temporal pattern recognition to predict future risk trajectories. This
          approach enables proactive intervention planning and personalized risk
          management strategies.
        </p>
        <p>
          The forecasting model incorporates both clinical parameters and
          biomarkers to generate predictions, with confidence intervals
          reflecting the inherent uncertainty in long-term forecasting.
        </p>
        <p>
          <strong>References:</strong>
        </p>
        <ol>
          <li>
            Cheng, L., et al. (2020). "Temporal Patterns Mining in Electronic
            Health Records using Deep Learning"
          </li>
          <li>
            Rajkomar, A., et al. (2022). "Machine Learning for Electronic Health
            Records"
          </li>
          <li>
            Goldstein, B.A., et al. (2021). "Opportunities and Challenges in
            Developing Risk Prediction Models with Electronic Health Records
            Data"
          </li>
          <li>
            Verma, S., et al. (2023). "Counterfactual Explanations for Machine
            Learning: A Review of Methods and Applications in Healthcare"
          </li>
        </ol>
      </div>
    </div>
  );
};

export default PatientForecast;
