import React, { useState } from "react";
import axios from "axios";
import "../styles/PatientForm.css";

/**
 * FollowUpVisitForm - A component for recording follow-up patient visits
 *
 * This component allows clinicians to record longitudinal patient data for research purposes,
 * focusing on changes in clinical parameters and biomarkers over time.
 *
 * References:
 * 1. Ibrahim JG, et al. (2010). "Missing Data in Clinical Studies: Issues and Methods"
 * 2. Diggle P, et al. (2002). "Analysis of Longitudinal Data"
 */
const FollowUpVisitForm = ({ patientId, patientData, onVisitAdded }) => {
  const [formData, setFormData] = useState({
    patient_id: patientId,
    visit_type: "follow-up",
    clinical_parameters: {
      blood_pressure: patientData?.blood_pressure || "",
      cholesterol: patientData?.cholesterol || "",
      fasting_blood_sugar: patientData?.fasting_blood_sugar || "",
      max_heart_rate: patientData?.max_heart_rate || "",
      exercise_induced_angina: patientData?.exercise_induced_angina || false,
      st_depression: patientData?.st_depression || "",
    },
    biomarkers: {
      nt_probnp: "",
    },
    timestamp: new Date().toISOString(),
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;

    // Handle nested properties using dot notation in name
    if (name.includes(".")) {
      const [parent, child] = name.split(".");
      setFormData({
        ...formData,
        [parent]: {
          ...formData[parent],
          [child]: type === "checkbox" ? checked : value,
        },
      });
    } else {
      setFormData({
        ...formData,
        [name]: type === "checkbox" ? checked : value,
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setSuccess("");

    try {
      // Validate form data
      if (!formData.visit_type) {
        throw new Error("Please select a visit type");
      }

      // Convert numeric fields to numbers
      const processedData = {
        ...formData,
        clinical_parameters: {
          ...formData.clinical_parameters,
          cholesterol: formData.clinical_parameters.cholesterol
            ? parseInt(formData.clinical_parameters.cholesterol)
            : null,
          fasting_blood_sugar: formData.clinical_parameters.fasting_blood_sugar
            ? parseInt(formData.clinical_parameters.fasting_blood_sugar)
            : null,
          max_heart_rate: formData.clinical_parameters.max_heart_rate
            ? parseInt(formData.clinical_parameters.max_heart_rate)
            : null,
          st_depression: formData.clinical_parameters.st_depression
            ? parseFloat(formData.clinical_parameters.st_depression)
            : null,
        },
        biomarkers: {
          ...formData.biomarkers,
          nt_probnp: formData.biomarkers.nt_probnp
            ? parseFloat(formData.biomarkers.nt_probnp)
            : null,
        },
      };

      console.log("Submitting follow-up visit data:", processedData);

      // Send data to the API
      const response = await axios.post(
        `http://localhost:8083/api/longitudinal/patients/${patientId}/visits`,
        processedData,
        {
          headers: {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
          },
          timeout: 10000, // 10 second timeout
        }
      );

      console.log("Follow-up visit response:", response.data);
      setSuccess("Follow-up visit recorded successfully!");

      // Reset form fields
      setFormData({
        ...formData,
        clinical_parameters: {
          ...formData.clinical_parameters,
          blood_pressure: "",
          cholesterol: "",
          fasting_blood_sugar: "",
          max_heart_rate: "",
          st_depression: "",
          exercise_induced_angina: false,
        },
        biomarkers: {
          ...formData.biomarkers,
          nt_probnp: "",
        },
      });

      // Notify parent component
      if (onVisitAdded) {
        // Check if NT-proBNP was added
        const ntProBNPAdded =
          formData.biomarkers.nt_probnp &&
          formData.biomarkers.nt_probnp.trim() !== "";

        // Log for debugging
        if (ntProBNPAdded) {
          console.log(
            "NT-proBNP value was added:",
            formData.biomarkers.nt_probnp
          );
        }

        onVisitAdded(response.data);
      }
    } catch (err) {
      console.error("Error recording follow-up visit:", err);
      setError(
        err.message ||
          err.response?.data?.error ||
          "An error occurred while recording the follow-up visit."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="follow-up-visit-form">
      <h2>Record Follow-Up Visit</h2>

      {error && (
        <div
          className="error-message"
          style={{ color: "red", marginBottom: "15px" }}
        >
          {error}
        </div>
      )}

      {success && (
        <div
          className="success-message"
          style={{ color: "green", marginBottom: "15px" }}
        >
          {success}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-section">
          <h3 className="form-section-title">Visit Information</h3>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Visit Type</label>
              <select
                name="visit_type"
                value={formData.visit_type}
                onChange={handleChange}
                className="form-input"
              >
                <option value="follow-up">Follow-up</option>
                <option value="emergency">Emergency</option>
                <option value="scheduled">Scheduled</option>
                <option value="telehealth">Telehealth</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Visit Date</label>
              <input
                type="datetime-local"
                name="timestamp"
                value={formData.timestamp.slice(0, 16)}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    timestamp: new Date(e.target.value).toISOString(),
                  })
                }
                className="form-input"
              />
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3 className="form-section-title">Clinical Parameters</h3>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Blood Pressure (mmHg)</label>
              <input
                type="text"
                name="clinical_parameters.blood_pressure"
                value={formData.clinical_parameters.blood_pressure}
                onChange={handleChange}
                className="form-input"
                placeholder="e.g., 120/80"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Cholesterol (mg/dL)</label>
              <input
                type="number"
                name="clinical_parameters.cholesterol"
                value={formData.clinical_parameters.cholesterol}
                onChange={handleChange}
                className="form-input"
                placeholder="e.g., 200"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Fasting Blood Sugar (mg/dL)</label>
              <input
                type="number"
                name="clinical_parameters.fasting_blood_sugar"
                value={formData.clinical_parameters.fasting_blood_sugar}
                onChange={handleChange}
                className="form-input"
                placeholder="e.g., 100"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Max Heart Rate (bpm)</label>
              <input
                type="number"
                name="clinical_parameters.max_heart_rate"
                value={formData.clinical_parameters.max_heart_rate}
                onChange={handleChange}
                className="form-input"
                placeholder="e.g., 75"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">ST Depression (mm)</label>
              <input
                type="number"
                name="clinical_parameters.st_depression"
                value={formData.clinical_parameters.st_depression}
                onChange={handleChange}
                className="form-input"
                step="0.1"
                placeholder="e.g., 0.2"
              />
            </div>

            <div className="form-group checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  name="clinical_parameters.exercise_induced_angina"
                  checked={formData.clinical_parameters.exercise_induced_angina}
                  onChange={handleChange}
                />
                Exercise-Induced Angina
              </label>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3 className="form-section-title">Biomarkers</h3>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">NT-proBNP (pg/mL)</label>
              <input
                type="number"
                name="biomarkers.nt_probnp"
                value={formData.biomarkers.nt_probnp}
                onChange={handleChange}
                className="form-input"
                placeholder="e.g., 125"
              />
              <small
                className="form-text"
                style={{
                  display: "block",
                  marginTop: "5px",
                  fontSize: "0.8rem",
                  color: "#666",
                }}
              >
                Reference range:{" "}
                {parseInt(patientData?.age) < 50
                  ? "0-450"
                  : parseInt(patientData?.age) <= 75
                  ? "0-900"
                  : "0-1800"}{" "}
                pg/mL (age-adjusted)
              </small>
            </div>
          </div>

          <div
            className="info-message"
            style={{ marginTop: "10px", color: "#666", fontStyle: "italic" }}
          >
            <strong>Research Note:</strong> Longitudinal NT-proBNP measurements
            provide valuable insights into cardiac stress over time. Changes of
            &gt;30% are considered clinically significant.
          </div>
        </div>

        <div className="form-actions">
          <button type="submit" className="submit-button" disabled={loading}>
            {loading ? "Recording..." : "Record Follow-Up Visit"}
          </button>
        </div>
      </form>
    </div>
  );
};

export default FollowUpVisitForm;
