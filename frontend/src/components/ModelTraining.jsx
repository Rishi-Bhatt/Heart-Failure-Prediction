import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/model-training.css";
import "../styles/model-training-stats.css";
import TrainingHistoryTable from "./TrainingHistoryTable";
import ModelTrainingStats from "./ModelTrainingStats";

// Import training configuration styles
import "../styles/model-training-config.css";

const ModelTraining = () => {
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [patientList, setPatientList] = useState([]);
  const [selectedPatient, setSelectedPatient] = useState("");
  const [feedbackValue, setFeedbackValue] = useState("1");
  const [isFeedbackSubmitting, setIsFeedbackSubmitting] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState(null);

  // Training configuration state
  const [trainingConfig, setTrainingConfig] = useState({
    epochs: 50,
    use_neural_network: false,
    batch_size: 32,
    validation_split: 0.2,
    early_stopping_patience: 10,
    learning_rate: 0.001,
    retraining_threshold: 20,
    drift_detection_threshold: 0.1,
    records_since_last_retraining: 0,
    last_retraining_date: null,
    retraining_count: 0,
  });
  const [configLoading, setConfigLoading] = useState(false);
  const [configMessage, setConfigMessage] = useState("");

  const navigate = useNavigate();

  // Define fetchTrainingHistory with useCallback
  const fetchTrainingHistory = useCallback(async () => {
    try {
      // Add cache-busting parameter
      const timestamp = new Date().getTime();
      const response = await fetch(
        `http://localhost:8083/api/retraining/history?t=${timestamp}`,
        {
          cache: "no-store", // Ensure no caching
          headers: {
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
            Expires: "0",
          },
        }
      );
      const data = await response.json();
      console.log("Fetched training history:", data);

      // Handle both array and object responses
      if (Array.isArray(data)) {
        // Sort by timestamp (newest first)
        const sortedData = [...data].sort(
          (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
        );
        console.log(`Sorted ${sortedData.length} training history entries`);
        setTrainingHistory(sortedData);
      } else {
        console.log(
          "Training history is not an array, wrapping in array:",
          data
        );
        setTrainingHistory([data]);
      }
      return true;
    } catch (error) {
      console.error("Error fetching training history:", error);
      setMessage("Failed to fetch training history");
      return false;
    }
  }, []);

  const fetchPatients = useCallback(async () => {
    try {
      // Add cache-busting parameter
      const timestamp = new Date().getTime();
      const response = await fetch(
        `http://localhost:8083/api/patients?t=${timestamp}`,
        {
          cache: "no-store", // Ensure no caching
          headers: {
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
            Expires: "0",
          },
        }
      );

      // Get response headers for debugging
      const patientCount = response.headers.get("x-patient-count") || "unknown";
      const responseTimestamp = response.headers.get("x-timestamp") || "none";

      const data = await response.json();
      console.log("Fetched patients for model training:", data);
      console.log(
        `Response headers - Count: ${patientCount}, Timestamp: ${responseTimestamp}`
      );

      // The response is now directly an array
      const patientsArray = data;
      console.log(
        `Received ${patientsArray.length} patients for model training directly as array`
      );

      // Process the data to ensure it has the correct structure
      if (Array.isArray(patientsArray)) {
        const processedData = patientsArray.map((patient) => {
          // Extract file modification time if available for debugging
          const fileInfo = patient.file_mtime
            ? ` (${new Date(patient.file_mtime * 1000).toLocaleString()})`
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

        setPatientList(processedData);
      } else {
        console.error("Patient list is not an array:", patientsArray);
        setPatientList([]); // Set to empty array to prevent errors
      }
      return true;
    } catch (error) {
      console.error("Error fetching patients:", error);
      setPatientList([]); // Set to empty array to prevent errors
      return false;
    }
  }, []);

  // Function to fetch training configuration
  const fetchTrainingConfig = useCallback(async () => {
    try {
      setConfigLoading(true);
      // Add cache-busting parameter
      const timestamp = new Date().getTime();
      const response = await fetch(
        `http://localhost:8083/api/training/config?t=${timestamp}`,
        {
          cache: "no-store", // Ensure no caching
          headers: {
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
            Expires: "0",
          },
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const config = await response.json();
      console.log("Fetched training configuration:", config);

      // Update state with fetched configuration
      setTrainingConfig((prev) => ({
        ...prev,
        ...config,
      }));

      return true;
    } catch (error) {
      console.error("Error fetching training configuration:", error);
      setConfigMessage("Failed to fetch training configuration");
      return false;
    } finally {
      setConfigLoading(false);
    }
  }, []);

  // Function to update training configuration
  const updateTrainingConfig = async (newConfig) => {
    try {
      setConfigLoading(true);
      setConfigMessage("");

      const response = await fetch(
        "http://localhost:8083/api/training/config",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(newConfig),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log("Configuration update result:", result);

      // Show success message
      setConfigMessage(result.message || "Configuration updated successfully");

      // Refresh configuration
      fetchTrainingConfig();

      return true;
    } catch (error) {
      console.error("Error updating configuration:", error);
      setConfigMessage(`Error updating configuration: ${error.message}`);
      return false;
    } finally {
      setConfigLoading(false);
    }
  };

  // Add useEffect after all fetch functions are defined
  useEffect(() => {
    // Fetch training history
    fetchTrainingHistory();

    // Fetch patient list
    fetchPatients();

    // Fetch training configuration
    fetchTrainingConfig();

    // Set up interval to refresh training config every 30 seconds
    const configIntervalId = setInterval(() => {
      console.log("Refreshing training configuration...");
      fetchTrainingConfig();
    }, 30000);

    // Set up polling interval for patient list (every 10 seconds)
    const intervalId = setInterval(() => {
      console.log("Polling for patient updates in model training...");
      fetchPatients();
    }, 10000);

    // Clean up intervals on component unmount
    return () => {
      clearInterval(intervalId);
      clearInterval(configIntervalId);
    };
  }, [fetchTrainingHistory, fetchPatients, fetchTrainingConfig]);

  const handleRetrain = async () => {
    setIsLoading(true);
    setMessage("");

    try {
      // Add timestamp to prevent caching
      const timestamp = new Date().getTime();
      const response = await fetch(
        `http://localhost:8083/api/retrain?t=${timestamp}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          // Include current training configuration with defaults for empty values
          body: JSON.stringify({
            epochs:
              trainingConfig.epochs === ""
                ? 50
                : parseInt(trainingConfig.epochs) || 50,
            use_neural_network: trainingConfig.use_neural_network,
            batch_size:
              trainingConfig.batch_size === ""
                ? 32
                : parseInt(trainingConfig.batch_size) || 32,
            validation_split:
              trainingConfig.validation_split === ""
                ? 0.2
                : parseFloat(trainingConfig.validation_split) || 0.2,
            early_stopping_patience:
              trainingConfig.early_stopping_patience === ""
                ? 10
                : parseInt(trainingConfig.early_stopping_patience) || 10,
            learning_rate:
              trainingConfig.learning_rate === ""
                ? 0.001
                : parseFloat(trainingConfig.learning_rate) || 0.001,
          }),
        }
      );

      const data = await response.json();
      console.log("Retrain response:", data);

      if (data.success) {
        setMessage(`Success: ${data.message}`);
        // Refresh training history and patient list after retraining
        fetchTrainingHistory();
        fetchPatients();
        // Also refresh the configuration
        fetchTrainingConfig();
      } else {
        setMessage(`Error: ${data.message || data.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Error retraining model:", error);
      setMessage("Failed to retrain model. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFeedbackSubmit = async (e) => {
    e.preventDefault();

    if (!selectedPatient) {
      setFeedbackMessage(
        <div className="feedback-error">Please select a patient</div>
      );
      return;
    }

    setIsFeedbackSubmitting(true);
    setFeedbackMessage(null);

    try {
      const response = await fetch("http://localhost:8083/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_id: selectedPatient,
          prediction: 0, // This will be overwritten by the backend with the actual prediction
          actual: parseInt(feedbackValue, 10),
        }),
      });

      const data = await response.json();

      if (data.success) {
        // Create a more detailed success message
        const predictionText =
          data.prediction >= 0.5
            ? "Heart Failure Risk"
            : "No Heart Failure Risk";
        const actualText =
          data.actual === 1 ? "Heart Failure Risk" : "No Heart Failure Risk";
        const correctnessText = data.is_correct ? "correct" : "incorrect";

        setFeedbackMessage(
          <div className="feedback-success">
            <p>
              <strong>Success:</strong> {data.message}
            </p>
            <p>
              The model predicted:{" "}
              <span
                className={data.prediction >= 0.5 ? "risk-high" : "risk-low"}
              >
                {predictionText} ({(data.prediction * 100).toFixed(1)}%)
              </span>
            </p>
            <p>
              You marked it as:{" "}
              <span className={data.actual === 1 ? "risk-high" : "risk-low"}>
                {actualText}
              </span>
            </p>
            <p>
              This feedback was recorded as <strong>{correctnessText}</strong>.
            </p>
          </div>
        );

        // Reset form
        setSelectedPatient("");
        setFeedbackValue("1");
      } else {
        setFeedbackMessage(
          <div className="feedback-error">Error: {data.message}</div>
        );
      }
    } catch (error) {
      console.error("Error submitting feedback:", error);
      setFeedbackMessage(
        <div className="feedback-error">
          Failed to submit feedback. Please try again.
        </div>
      );
    } finally {
      setIsFeedbackSubmitting(false);
    }
  };

  return (
    <div className="model-training-container">
      <h2>Model Training and Feedback</h2>

      <ModelTrainingStats />

      <div className="training-section">
        <h3>Model Retraining</h3>
        <p>
          Retrain the model with all available patient data and feedback to
          improve prediction accuracy.
        </p>

        {/* Training Configuration Section */}
        <div className="training-config-section">
          <h4>Training Configuration</h4>
          <div className="config-form">
            <div className="config-row">
              <div className="config-item">
                <label htmlFor="use_neural_network">Use Neural Network:</label>
                <div className="toggle-switch">
                  <input
                    type="checkbox"
                    id="use_neural_network"
                    checked={trainingConfig.use_neural_network}
                    onChange={(e) => {
                      const isChecked = e.target.checked;
                      console.log("Toggle changed to:", isChecked);
                      setTrainingConfig((prev) => ({
                        ...prev,
                        use_neural_network: isChecked,
                      }));
                    }}
                  />
                  <label
                    htmlFor="use_neural_network"
                    className="slider"
                  ></label>
                </div>
              </div>

              <div className="config-item">
                <label htmlFor="epochs">Training Epochs:</label>
                <input
                  type="number"
                  id="epochs"
                  min="1"
                  max="1000"
                  value={trainingConfig.epochs}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({
                      ...prev,
                      epochs: value === "" ? "" : parseInt(value) || 50,
                    }));
                  }}
                />
              </div>
            </div>

            <div className="config-row">
              <div className="config-item">
                <label htmlFor="batch_size">Batch Size:</label>
                <input
                  type="number"
                  id="batch_size"
                  min="1"
                  max="256"
                  value={trainingConfig.batch_size}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({
                      ...prev,
                      batch_size: value === "" ? "" : parseInt(value) || 32,
                    }));
                  }}
                />
              </div>

              <div className="config-item">
                <label htmlFor="validation_split">Validation Split:</label>
                <input
                  type="number"
                  id="validation_split"
                  min="0.1"
                  max="0.5"
                  step="0.05"
                  value={trainingConfig.validation_split}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({
                      ...prev,
                      validation_split:
                        value === "" ? "" : parseFloat(value) || 0.2,
                    }));
                  }}
                />
              </div>
            </div>

            <div className="config-row">
              <div className="config-item">
                <label htmlFor="early_stopping_patience">
                  Early Stopping Patience:
                </label>
                <input
                  type="number"
                  id="early_stopping_patience"
                  min="1"
                  max="50"
                  value={trainingConfig.early_stopping_patience}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({
                      ...prev,
                      early_stopping_patience:
                        value === "" ? "" : parseInt(value) || 10,
                    }));
                  }}
                />
              </div>

              <div className="config-item">
                <label htmlFor="learning_rate">Learning Rate:</label>
                <input
                  type="number"
                  id="learning_rate"
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                  value={trainingConfig.learning_rate}
                  onChange={(e) => {
                    const value = e.target.value;
                    setTrainingConfig((prev) => ({
                      ...prev,
                      learning_rate:
                        value === "" ? "" : parseFloat(value) || 0.001,
                    }));
                  }}
                />
              </div>
            </div>

            <div className="config-actions">
              <button
                className="save-config-button"
                onClick={() => {
                  // Convert empty string values to defaults before saving
                  const configToSave = {
                    ...trainingConfig,
                    epochs:
                      trainingConfig.epochs === ""
                        ? 50
                        : parseInt(trainingConfig.epochs) || 50,
                    batch_size:
                      trainingConfig.batch_size === ""
                        ? 32
                        : parseInt(trainingConfig.batch_size) || 32,
                    validation_split:
                      trainingConfig.validation_split === ""
                        ? 0.2
                        : parseFloat(trainingConfig.validation_split) || 0.2,
                    early_stopping_patience:
                      trainingConfig.early_stopping_patience === ""
                        ? 10
                        : parseInt(trainingConfig.early_stopping_patience) ||
                          10,
                    learning_rate:
                      trainingConfig.learning_rate === ""
                        ? 0.001
                        : parseFloat(trainingConfig.learning_rate) || 0.001,
                  };
                  updateTrainingConfig(configToSave);
                }}
                disabled={configLoading}
              >
                {configLoading ? "Saving..." : "Save Configuration"}
              </button>

              {configMessage && (
                <div
                  className={
                    configMessage.includes("Error")
                      ? "error-message"
                      : "success-message"
                  }
                >
                  {configMessage}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="retraining-action">
          <button
            className="retrain-button"
            onClick={handleRetrain}
            disabled={isLoading}
          >
            {isLoading ? "Retraining..." : "Retrain Model"}
          </button>
          {message && (
            <div
              className={
                message.includes("Error") ? "error-message" : "success-message"
              }
            >
              {message}
            </div>
          )}
        </div>
      </div>

      <div className="feedback-section">
        <h3>Provide Feedback</h3>
        <p>
          Provide feedback on predictions to help improve the model's accuracy.
        </p>
        <form onSubmit={handleFeedbackSubmit}>
          <div className="form-group">
            <label htmlFor="patient-select">Select Patient:</label>
            <select
              id="patient-select"
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              required
            >
              <option value="">-- Select a patient --</option>
              {Array.isArray(patientList) ? (
                patientList.map((patient) => (
                  <option key={patient.patient_id} value={patient.patient_id}>
                    {patient.name} (ID: {patient.patient_id})
                  </option>
                ))
              ) : (
                <option value="" disabled>
                  No patients available
                </option>
              )}
            </select>
          </div>

          <div className="form-group">
            <label>Actual Outcome:</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="feedback"
                  value="1"
                  checked={feedbackValue === "1"}
                  onChange={() => setFeedbackValue("1")}
                />
                Heart Failure Risk
              </label>
              <label>
                <input
                  type="radio"
                  name="feedback"
                  value="0"
                  checked={feedbackValue === "0"}
                  onChange={() => setFeedbackValue("0")}
                />
                No Heart Failure Risk
              </label>
            </div>
          </div>

          <button
            type="submit"
            className="submit-button"
            disabled={isFeedbackSubmitting}
          >
            {isFeedbackSubmitting ? "Submitting..." : "Submit Feedback"}
          </button>
          {feedbackMessage &&
            (typeof feedbackMessage === "string" ? (
              <div
                className={
                  feedbackMessage.includes("Error")
                    ? "error-message"
                    : "success-message"
                }
              >
                {feedbackMessage}
              </div>
            ) : (
              // If feedbackMessage is a JSX element, render it directly
              feedbackMessage
            ))}
        </form>
      </div>

      <div className="history-section">
        <TrainingHistoryTable trainingHistory={trainingHistory} />
      </div>

      <div className="navigation-buttons">
        <button onClick={() => navigate("/")}>Back to Home</button>
        <button onClick={() => navigate("/history")}>
          View Patient History
        </button>
      </div>
    </div>
  );
};

export default ModelTraining;
