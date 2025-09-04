import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import BiomarkerExplanation from "./BiomarkerExplanation";
import EnhancedBiomarkers from "./EnhancedBiomarkers";
import EnhancedMedications from "./EnhancedMedications";

const PatientForm = ({ setPredictionResult }) => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [useEnhancedFeatures, setUseEnhancedFeatures] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "Male",
    blood_pressure: "",
    cholesterol: "",
    fasting_blood_sugar: "",
    chest_pain_type: "Typical Angina",
    ecg_result: "Normal",
    max_heart_rate: "",
    exercise_induced_angina: false,
    st_depression: "",
    slope_of_st: "Flat",
    number_of_major_vessels: 0,
    thalassemia: "Normal",
    prior_cardiac_event: {
      type: "",
      time_since_event: "",
      severity: "Mild",
    },
    biomarkers: {
      nt_probnp: "",
      troponin: "",
      crp: "",
      bnp: "",
      creatinine: "",
    },
    medications: [],
    // New medication format (for enhanced features)
    medicationsNew: {
      ace_inhibitor: false,
      arb: false,
      beta_blocker: false,
      statin: false,
      antiplatelet: false,
      diuretic: false,
      calcium_channel_blocker: false,
    },
  });

  const [medication, setMedication] = useState({
    type: "Beta-blockers",
    time_of_administration: "",
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;

    if (name.includes(".")) {
      // Handle nested objects (prior_cardiac_event)
      const [parent, child] = name.split(".");
      setFormData({
        ...formData,
        [parent]: {
          ...formData[parent],
          [child]: type === "checkbox" ? checked : value,
        },
      });
    } else {
      // Handle regular inputs
      setFormData({
        ...formData,
        [name]: type === "checkbox" ? checked : value,
      });
    }
  };

  const handleMedicationChange = (e) => {
    const { name, value } = e.target;
    setMedication({
      ...medication,
      [name]: value,
    });
  };

  const addMedication = () => {
    if (medication.type && medication.time_of_administration) {
      setFormData({
        ...formData,
        medications: [...formData.medications, { ...medication }],
      });
      setMedication({
        type: "Beta-blockers",
        time_of_administration: "",
      });
    }
  };

  const removeMedication = (index) => {
    const updatedMedications = [...formData.medications];
    updatedMedications.splice(index, 1);
    setFormData({
      ...formData,
      medications: updatedMedications,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Create a copy of the form data
      const submissionData = { ...formData };

      // Handle medications based on the mode
      if (useEnhancedFeatures) {
        // Use the new medication format
        submissionData.medications = submissionData.medicationsNew;
        delete submissionData.medicationsNew;
      }

      // Add a timestamp to prevent caching
      const timestamp = new Date().getTime();
      const response = await axios.post(
        `http://localhost:8083/api/predict?t=${timestamp}`,
        submissionData,
        {
          headers: {
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
            Expires: "0",
          },
        }
      );

      console.log("Prediction response:", response.data);
      setPredictionResult(response.data);

      // Navigate to results page
      navigate("/results");

      // Don't automatically redirect to history page
      // This gives the user time to view the results
    } catch (error) {
      console.error("Error submitting form:", error);
      alert(
        "An error occurred while processing your request. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="form-container">
      <h2 className="form-title">Patient Information</h2>

      {/* Enhanced Features Toggle */}
      <div
        className="enhanced-features-toggle"
        style={{ marginBottom: "20px" }}
      >
        <label
          style={{ display: "flex", alignItems: "center", cursor: "pointer" }}
        >
          <input
            type="checkbox"
            checked={useEnhancedFeatures}
            onChange={() => setUseEnhancedFeatures(!useEnhancedFeatures)}
            style={{ marginRight: "10px" }}
          />
          <span style={{ fontWeight: "500" }}>
            Use Enhanced Features (NT-proBNP, Medications, Prior Events)
          </span>
        </label>
        <p
          style={{ margin: "5px 0 0 25px", fontSize: "0.9rem", color: "#666" }}
        >
          Enables advanced biomarkers and medication tracking for more accurate
          predictions
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Basic Information */}
        <div className="form-section">
          <h3 className="form-section-title">Basic Information</h3>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Name</label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                className="form-input"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Age</label>
              <input
                type="number"
                name="age"
                value={formData.age}
                onChange={handleChange}
                className="form-input"
                min="18"
                max="120"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Gender</label>
              <select
                name="gender"
                value={formData.gender}
                onChange={handleChange}
                className="form-select"
                required
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>
        </div>

        {/* Clinical Measurements */}
        <div className="form-section">
          <h3 className="form-section-title">Clinical Measurements</h3>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">
                Blood Pressure (e.g., 140/90)
              </label>
              <input
                type="text"
                name="blood_pressure"
                value={formData.blood_pressure}
                onChange={handleChange}
                className="form-input"
                placeholder="e.g., 140/90"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Cholesterol (mg/dL)</label>
              <input
                type="number"
                name="cholesterol"
                value={formData.cholesterol}
                onChange={handleChange}
                className="form-input"
                min="100"
                max="500"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Fasting Blood Sugar (mg/dL)</label>
              <input
                type="number"
                name="fasting_blood_sugar"
                value={formData.fasting_blood_sugar}
                onChange={handleChange}
                className="form-input"
                min="50"
                max="300"
                required
              />
            </div>
          </div>
        </div>

        {/* Cardiac Assessment */}
        <div className="form-section">
          <h3 className="form-section-title">Cardiac Assessment</h3>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Chest Pain Type</label>
              <select
                name="chest_pain_type"
                value={formData.chest_pain_type}
                onChange={handleChange}
                className="form-select"
                required
              >
                <option value="Typical Angina">Typical Angina</option>
                <option value="Atypical Angina">Atypical Angina</option>
                <option value="Non-Anginal Pain">Non-Anginal Pain</option>
                <option value="Asymptomatic">Asymptomatic</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">ECG Result</label>
              <select
                name="ecg_result"
                value={formData.ecg_result}
                onChange={handleChange}
                className="form-select"
                required
              >
                <option value="Normal">Normal</option>
                <option value="ST-T Wave Abnormality">
                  ST-T Wave Abnormality
                </option>
                <option value="Left Ventricular Hypertrophy">
                  Left Ventricular Hypertrophy
                </option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Max Heart Rate</label>
              <input
                type="number"
                name="max_heart_rate"
                value={formData.max_heart_rate}
                onChange={handleChange}
                className="form-input"
                min="60"
                max="220"
                required
              />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Exercise Induced Angina</label>
              <input
                type="checkbox"
                name="exercise_induced_angina"
                checked={formData.exercise_induced_angina}
                onChange={handleChange}
              />
            </div>
            <div className="form-group">
              <label className="form-label">ST Depression</label>
              <input
                type="number"
                name="st_depression"
                value={formData.st_depression}
                onChange={handleChange}
                className="form-input"
                step="0.1"
                min="0"
                max="10"
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Slope of ST</label>
              <select
                name="slope_of_st"
                value={formData.slope_of_st}
                onChange={handleChange}
                className="form-select"
                required
              >
                <option value="Upsloping">Upsloping</option>
                <option value="Flat">Flat</option>
                <option value="Downsloping">Downsloping</option>
              </select>
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Number of Major Vessels</label>
              <select
                name="number_of_major_vessels"
                value={formData.number_of_major_vessels}
                onChange={handleChange}
                className="form-select"
                required
              >
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Thalassemia</label>
              <select
                name="thalassemia"
                value={formData.thalassemia}
                onChange={handleChange}
                className="form-select"
                required
              >
                <option value="Normal">Normal</option>
                <option value="Fixed Defect">Fixed Defect</option>
                <option value="Reversible Defect">Reversible Defect</option>
              </select>
            </div>
          </div>
        </div>

        {/* Prior Cardiac Event */}
        <div className="form-section">
          <h3 className="form-section-title">Prior Cardiac Event</h3>
          {!formData.prior_cardiac_event.type && (
            <div
              className="info-message"
              style={{
                marginBottom: "15px",
                color: "#666",
                fontStyle: "italic",
              }}
            >
              Select a cardiac event type if the patient has a history of
              heart-related conditions.
            </div>
          )}
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Type</label>
              <select
                name="prior_cardiac_event.type"
                value={formData.prior_cardiac_event.type}
                onChange={handleChange}
                className="form-select"
              >
                <option value="">None</option>
                <option value="Myocardial Infarction">
                  Myocardial Infarction (Heart Attack)
                </option>
                <option value="Arrhythmia">Arrhythmia</option>
                <option value="Angina">Angina</option>
                <option value="Heart Failure">Heart Failure</option>
                <option value="Coronary Artery Disease">
                  Coronary Artery Disease
                </option>
                <option value="Valve Disease">Valve Disease</option>
                <option value="Cardiomyopathy">Cardiomyopathy</option>
                <option value="Pericarditis">Pericarditis</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Time Since Event (months)</label>
              <select
                name="prior_cardiac_event.time_since_event"
                value={formData.prior_cardiac_event.time_since_event}
                onChange={handleChange}
                className="form-select"
                disabled={!formData.prior_cardiac_event.type}
              >
                <option value="">Select time</option>
                <option value="1">Less than 1 month</option>
                <option value="3">1-3 months</option>
                <option value="6">3-6 months</option>
                <option value="12">6-12 months</option>
                <option value="24">1-2 years</option>
                <option value="36">2-3 years</option>
                <option value="60">3-5 years</option>
                <option value="120">More than 5 years</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Severity</label>
              <select
                name="prior_cardiac_event.severity"
                value={formData.prior_cardiac_event.severity}
                onChange={handleChange}
                className="form-select"
                disabled={!formData.prior_cardiac_event.type}
              >
                <option value="Mild">Mild</option>
                <option value="Moderate">Moderate</option>
                <option value="Severe">Severe</option>
              </select>
            </div>
          </div>
        </div>

        {/* Biomarkers */}
        <div className="form-section">
          <h3 className="form-section-title">Biomarkers (Optional)</h3>
          <div
            className="info-message"
            style={{ marginBottom: "15px", color: "#666", fontStyle: "italic" }}
          >
            Enter biomarker values if available from lab tests. NT-proBNP is a
            cardiac biomarker that can significantly improve prediction accuracy
            (up to 15% improvement).
          </div>

          {useEnhancedFeatures ? (
            <EnhancedBiomarkers
              biomarkers={formData.biomarkers}
              handleChange={handleChange}
              age={formData.age}
            />
          ) : (
            <div className="form-row">
              <div className="form-group">
                <label className="form-label">NT-proBNP (pg/mL)</label>
                <input
                  type="number"
                  name="biomarkers.nt_probnp"
                  value={formData.biomarkers?.nt_probnp || ""}
                  onChange={handleChange}
                  className="form-input"
                  min="0"
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
                  {parseInt(formData.age) < 50
                    ? "0-450"
                    : parseInt(formData.age) <= 75
                    ? "0-900"
                    : "0-1800"}{" "}
                  pg/mL (age-adjusted)
                </small>
              </div>
            </div>
          )}

          {/* Scientific explanation toggle */}
          <div style={{ marginTop: "15px" }}>
            <details>
              <summary
                style={{
                  cursor: "pointer",
                  color: "#3498db",
                  fontWeight: "500",
                }}
              >
                View Scientific Information on NT-proBNP
              </summary>
              <div style={{ marginTop: "15px" }}>
                <BiomarkerExplanation />
              </div>
            </details>
          </div>
        </div>

        {/* Medications */}
        <div className="form-section">
          <h3 className="form-section-title">Medications</h3>

          {useEnhancedFeatures ? (
            <div>
              <div
                className="info-message"
                style={{
                  marginBottom: "15px",
                  color: "#666",
                  fontStyle: "italic",
                }}
              >
                Select all medications the patient is currently taking. These
                will be factored into the risk assessment.
              </div>
              <EnhancedMedications
                medications={formData.medicationsNew}
                handleChange={handleChange}
              />
            </div>
          ) : (
            <div>
              <h4 className="form-subtitle">Medications (first 24 hours)</h4>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Type</label>
                  <select
                    name="type"
                    value={medication.type}
                    onChange={handleMedicationChange}
                    className="form-select"
                  >
                    <option value="Beta-blockers">Beta-blockers</option>
                    <option value="ACE inhibitors">ACE inhibitors</option>
                    <option value="Aspirin">Aspirin</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">
                    Time of Administration (hours post-event)
                  </label>
                  <input
                    type="number"
                    name="time_of_administration"
                    value={medication.time_of_administration}
                    onChange={handleMedicationChange}
                    className="form-input"
                    min="0"
                    max="24"
                  />
                </div>
                <div
                  className="form-group"
                  style={{ display: "flex", alignItems: "flex-end" }}
                >
                  <button
                    type="button"
                    onClick={addMedication}
                    className="form-button"
                    style={{ marginTop: "1.5rem" }}
                  >
                    Add Medication
                  </button>
                </div>
              </div>

              {/* Medications List */}
              {formData.medications.length > 0 && (
                <div className="medications-list">
                  <h4>Added Medications:</h4>
                  <ul>
                    {formData.medications.map((med, index) => (
                      <li key={index}>
                        {med.type} - {med.time_of_administration} hours
                        post-event
                        <button
                          type="button"
                          onClick={() => removeMedication(index)}
                          style={{ marginLeft: "10px", color: "red" }}
                        >
                          Remove
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Submit Button */}
        <div
          className="form-row"
          style={{ justifyContent: "center", marginTop: "2rem" }}
        >
          <button type="submit" className="form-button" disabled={loading}>
            {loading ? "Processing..." : "Generate Prediction"}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PatientForm;
