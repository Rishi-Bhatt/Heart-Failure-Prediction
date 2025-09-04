import { useState, useEffect } from "react";
import axios from "axios";
import TrainingHistoryTable from "./TrainingHistoryTable";
import ModelTrainingStats from "./ModelTrainingStats";
import "../styles/model-training-stats.css";

const ModelRetraining = () => {
  const [retrainingHistory, setRetrainingHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [error, setError] = useState(null);

  // Fetch retraining history when component mounts
  useEffect(() => {
    fetchRetrainingHistory();
  }, []);

  // Function to fetch retraining history
  const fetchRetrainingHistory = async () => {
    try {
      setLoading(true);
      // Add cache-busting parameter
      const timestamp = new Date().getTime();
      const response = await axios.get(
        `http://localhost:8083/api/retraining/history?t=${timestamp}`,
        {
          headers: {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
            Expires: "0",
          },
        }
      );
      console.log("Fetched retraining history:", response.data);

      // Handle both array and object responses
      if (Array.isArray(response.data)) {
        // Sort by timestamp (newest first)
        const sortedData = [...response.data].sort(
          (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
        );
        console.log(`Sorted ${sortedData.length} retraining history entries`);
        setRetrainingHistory(sortedData);
      } else {
        console.log("Retraining history is not an array:", response.data);
        setRetrainingHistory(response.data);
      }

      setError(null);
    } catch (err) {
      console.error("Error fetching retraining history:", err);
      setError("Failed to load retraining history");
    } finally {
      setLoading(false);
    }
  };

  // Function to trigger model retraining
  const handleRetrain = async () => {
    try {
      setRetraining(true);
      const response = await axios.post(
        "http://localhost:8083/api/retrain",
        {},
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (response.data.success) {
        // Update retraining history after successful retraining
        setRetrainingHistory(response.data);
        setError(null);
        alert("Model retrained successfully!");
        // Refresh the retraining history
        fetchRetrainingHistory();
      } else {
        setError("Retraining failed: " + response.data.error);
      }
    } catch (err) {
      console.error("Error retraining model:", err);
      setError(
        "Failed to retrain model: " + (err.response?.data?.error || err.message)
      );
    } finally {
      setRetraining(false);
    }
  };

  // Format date for display
  const formatDate = (dateString) => {
    if (!dateString) return "Never";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  if (loading) {
    return <div>Loading retraining information...</div>;
  }

  return (
    <div className="form-container">
      <h2 className="form-title">Model Retraining</h2>

      <ModelTrainingStats />

      {error && (
        <div
          className="error-message"
          style={{ color: "red", marginBottom: "1rem" }}
        >
          {error}
        </div>
      )}

      <div className="retraining-info">
        <h3>Retraining Summary</h3>

        <div className="info-row">
          <div className="info-label">Last Retraining:</div>
          <div className="info-value">
            {formatDate(retrainingHistory?.timestamp)}
          </div>
        </div>

        <div className="info-row">
          <div className="info-label">Records Used:</div>
          <div className="info-value">
            {retrainingHistory?.num_records || 0}
          </div>
        </div>

        <div className="info-row">
          <div className="info-label">Total Records:</div>
          <div className="info-value">
            {retrainingHistory?.total_records ||
              retrainingHistory?.num_records ||
              0}
          </div>
        </div>

        <div className="info-row">
          <div className="info-label">Model Accuracy:</div>
          <div className="info-value">
            {retrainingHistory?.accuracy
              ? `${(retrainingHistory.accuracy * 100).toFixed(2)}%`
              : "Unknown"}
          </div>
        </div>

        {retrainingHistory?.message && (
          <div className="info-row">
            <div className="info-label">Status:</div>
            <div className="info-value">{retrainingHistory.message}</div>
          </div>
        )}
      </div>

      <div className="retraining-history">
        {Array.isArray(retrainingHistory) ? (
          <TrainingHistoryTable trainingHistory={retrainingHistory} />
        ) : retrainingHistory ? (
          <TrainingHistoryTable trainingHistory={[retrainingHistory]} />
        ) : null}
      </div>

      <div className="retraining-actions" style={{ marginTop: "2rem" }}>
        <button
          className="form-button"
          onClick={handleRetrain}
          disabled={retraining}
          style={{
            backgroundColor: retraining ? "#cccccc" : "#3498db",
            cursor: retraining ? "not-allowed" : "pointer",
          }}
        >
          {retraining ? "Retraining..." : "Retrain Model"}
        </button>

        <p style={{ marginTop: "1rem", fontSize: "0.9rem", color: "#666" }}>
          Retraining the model will use all previously collected patient data to
          improve prediction accuracy. This process may take a few moments to
          complete.
        </p>
      </div>
    </div>
  );
};

export default ModelRetraining;
