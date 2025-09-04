import React from 'react';

const TrainingHistoryTable = ({ trainingHistory }) => {
  if (!trainingHistory || trainingHistory.length === 0) {
    return <p>No training history available</p>;
  }

  return (
    <div className="training-history-container">
      <h3>Training History</h3>
      <table className="history-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Records Used</th>
            <th>Total Records</th>
            <th>Results</th>
          </tr>
        </thead>
        <tbody>
          {trainingHistory.map((entry, index) => (
            <tr key={index}>
              <td>{new Date(entry.timestamp).toLocaleString()}</td>
              <td>{entry.num_records}</td>
              <td>{entry.total_records || entry.num_records}</td>
              <td>
                {entry.metrics ? (
                  <div>
                    <div>
                      Accuracy: {(entry.metrics.accuracy * 100).toFixed(2)}%
                    </div>
                    {entry.metrics.precision && (
                      <div>
                        Precision: {(entry.metrics.precision * 100).toFixed(2)}%
                      </div>
                    )}
                    {entry.metrics.recall && (
                      <div>
                        Recall: {(entry.metrics.recall * 100).toFixed(2)}%
                      </div>
                    )}
                  </div>
                ) : entry.weights ? (
                  <div>
                    <div>Updated feature weights</div>
                    <button
                      className="view-weights-button"
                      onClick={() =>
                        alert(JSON.stringify(entry.weights, null, 2))
                      }
                    >
                      View Weights
                    </button>
                  </div>
                ) : (
                  <div>Completed successfully</div>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TrainingHistoryTable;
