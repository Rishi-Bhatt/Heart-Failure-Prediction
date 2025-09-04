import { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";

// Import components
import PatientForm from "./components/PatientForm";
import ResultsDisplay from "./components/ResultsDisplay";
import PatientHistory from "./components/PatientHistory";
import PatientHistoryManager from "./components/PatientHistoryManager";
import NewPatientHistory from "./components/NewPatientHistory";
import PatientDetail from "./components/PatientDetail";
import PatientForecast from "./components/PatientForecast";
import PatientTrajectoryPage from "./components/PatientTrajectoryPage";
import ModelTraining from "./components/ModelTraining";
import Navbar from "./components/Navbar";

// Import styles
import "./styles/patient-history-manager.css";
import "./styles/patient-detail.css";
import "./styles/new-patient-history.css";
import "./styles/abnormality-timeline.css";

function App() {
  const [predictionResult, setPredictionResult] = useState(null);

  return (
    <Router>
      <div className="app-container">
        <Navbar />
        <div className="content">
          <Routes>
            <Route
              path="/"
              element={
                <PatientForm setPredictionResult={setPredictionResult} />
              }
            />
            <Route
              path="/results"
              element={<ResultsDisplay predictionResult={predictionResult} />}
            />
            <Route path="/history" element={<NewPatientHistory />} />
            <Route path="/patients/:patientId" element={<PatientDetail />} />
            <Route
              path="/patients/:patientId/forecast"
              element={<PatientForecast />}
            />
            <Route
              path="/patients/:patientId/trajectory"
              element={<PatientTrajectoryPage />}
            />
            <Route path="/retrain" element={<ModelTraining />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
