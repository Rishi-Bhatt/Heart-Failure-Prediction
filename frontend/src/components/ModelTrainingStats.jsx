import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ModelTrainingStats = () => {
  const [patientCount, setPatientCount] = useState(0);
  const [trainingStats, setTrainingStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Fetch data when component mounts
    fetchData();

    // Set up polling interval (every 10 seconds)
    const intervalId = setInterval(() => {
      fetchData(false);
    }, 10000);

    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);

  const fetchData = async (showLoading = true) => {
    if (showLoading) {
      setLoading(true);
    }

    try {
      // Add timestamp to prevent caching
      const timestamp = new Date().getTime();
      
      // Fetch patient count
      const patientsResponse = await axios.get(
        `http://localhost:8083/api/patients?t=${timestamp}`,
        {
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Expires': '0'
          }
        }
      );
      
      // Fetch latest training history
      const trainingResponse = await axios.get(
        `http://localhost:8083/api/retraining/history?t=${timestamp}`,
        {
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Expires': '0'
          }
        }
      );

      // Update state with fetched data
      if (Array.isArray(patientsResponse.data)) {
        setPatientCount(patientsResponse.data.length);
        console.log(`Found ${patientsResponse.data.length} patients`);
      }

      if (Array.isArray(trainingResponse.data) && trainingResponse.data.length > 0) {
        // Sort by timestamp (newest first)
        const sortedData = [...trainingResponse.data].sort(
          (a, b) => new Date(b.timestamp) - new Date(a.timestamp)
        );
        setTrainingStats(sortedData[0]); // Get the most recent training event
        console.log('Latest training stats:', sortedData[0]);
      } else if (trainingResponse.data) {
        setTrainingStats(trainingResponse.data);
        console.log('Training stats:', trainingResponse.data);
      }

      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      if (showLoading) {
        setError('Failed to load data');
      }
    } finally {
      if (showLoading) {
        setLoading(false);
      }
    }
  };

  if (loading) {
    return <div className="stats-loading">Loading stats...</div>;
  }

  if (error) {
    return <div className="stats-error">{error}</div>;
  }

  return (
    <div className="model-training-stats">
      <h3>System Statistics</h3>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-title">Total Patients</div>
          <div className="stat-value">{patientCount}</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-title">Records Used in Training</div>
          <div className="stat-value">{trainingStats?.num_records || 0}</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-title">Total Records Available</div>
          <div className="stat-value">{trainingStats?.total_records || patientCount || 0}</div>
        </div>
        
        <div className="stat-card">
          <div className="stat-title">Model Accuracy</div>
          <div className="stat-value">
            {trainingStats?.metrics?.accuracy 
              ? `${(trainingStats.metrics.accuracy * 100).toFixed(1)}%` 
              : 'N/A'}
          </div>
        </div>
      </div>
      
      <div className="stats-footer">
        <button onClick={() => fetchData(true)} className="refresh-button">
          Refresh Stats
        </button>
        <div className="last-updated">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default ModelTrainingStats;
