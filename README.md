# Heart Failure Prediction System

A full-stack system that predicts heart failure risk using synthetic ECG signals and simulated clinical data.

## Features

- **Patient Data Input**: Form to input patient clinical data, prior cardiac events, and medications
- **Synthetic ECG Generation**: Backend generates ECG signals based on patient data
- **ECG Analysis**: Detects abnormalities like PVCs, flatlines, tachycardia, etc.
- **ML Prediction**: Random Forest model trained on Heart Failure Clinical Records Dataset
- **Explainable AI**: SHAP values for model explainability
- **Visualization**: ECG waveform with abnormality timeline
- **Model Retraining**: Automatic retraining after N records with drift detection

## Tech Stack

- **Frontend**: React with Vite, Chart.js for visualization
- **Backend**: Flask (Python)
- **ML**: Scikit-learn, XGBoost, SHAP
- **ECG Generation**: NeuroKit2
- **Data Storage**: Local JSON files

## Project Structure

```
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── data/                  # Data storage directory
│   ├── models/                # ML model directory
│   │   └── heart_failure_model.py  # Heart failure prediction model
│   ├── retraining/            # Model retraining utilities
│   │   └── model_retrainer.py # Model retraining logic
│   └── utils/                 # Utility functions
│       └── ecg_generator.py   # ECG generation and analysis
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── Navbar.jsx     # Navigation bar
│   │   │   ├── PatientForm.jsx # Patient data input form
│   │   │   ├── ResultsDisplay.jsx # Results visualization
│   │   │   ├── PatientHistory.jsx # Patient records list
│   │   │   └── PatientDetail.jsx # Individual patient details
│   │   ├── App.jsx            # Main React component
│   │   └── main.jsx           # React entry point
└── run.sh                     # Script to run both frontend and backend
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Environment Setup

#### Backend Setup

1. Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

2. Install backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Verify the installation:

```bash
python verify_installation.py
```

#### Frontend Setup

1. Install frontend dependencies:

```bash
cd frontend
npm install
```

### Running the Application

#### Option 1: Using the setup script

Run the setup script which will set up the environment and start both servers:

```bash
./run_with_setup.sh
```

#### Option 2: Manual startup

1. Start the backend server:

```bash
cd backend
python app.py
```

2. In a separate terminal, start the frontend server:

```bash
cd frontend
npm run dev
```

### Troubleshooting

If you encounter any issues during setup or running the application, please refer to the `TROUBLESHOOTING.md` file for common problems and solutions.

### Usage

1. Open your browser and navigate to `http://localhost:5173`
2. Fill out the patient form with clinical data
3. Submit to generate a prediction
4. View the results, including ECG visualization and risk assessment

## Data Simulation

All data in this system is simulated:

- Patient clinical data is generated based on patterns from the Heart Failure Clinical Records Dataset
- ECG signals are synthetically generated using NeuroKit2
- ECG abnormalities are simulated based on clinical parameters

## Model Retraining

The system includes automatic model retraining:

- Retrains after every 20 new patient records
- Includes drift detection to identify when model performance degrades
- Prioritizes misclassified samples in retraining

## License

This project is licensed under the MIT License - see the LICENSE file for details.
