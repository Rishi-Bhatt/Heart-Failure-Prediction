from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime

# Import simplified modules
from utils.simple_ecg_generator import generate_simple_ecg, analyze_simple_ecg
from models.simple_model import SimpleHeartFailureModel

app = Flask(__name__)
CORS(app)

# Initialize model
model = SimpleHeartFailureModel()

# Ensure data directories exist
os.makedirs('data/patients', exist_ok=True)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict heart failure risk based on patient data
    """
    try:
        print("Received prediction request")
        # Get patient data from request
        patient_data = request.json
        print(f"Patient data: {patient_data}")

        # Generate synthetic ECG based on patient data
        ecg_signal, ecg_time = generate_simple_ecg(patient_data)

        # Analyze ECG for abnormalities
        abnormalities = analyze_simple_ecg(ecg_signal, ecg_time, patient_data)

        # Make prediction (using dummy features)
        features = np.array([[
            patient_data.get('age', 60),
            1 if patient_data.get('gender', 'Male') == 'Male' else 0,
            patient_data.get('cholesterol', 200),
            patient_data.get('max_heart_rate', 75)
        ]])
        prediction, confidence, shap_values = model.predict(features)

        # Save patient data and prediction
        patient_id = save_patient_data(patient_data, ecg_signal, ecg_time,
                                      abnormalities, prediction, confidence, shap_values)

        print(f"Prediction complete: {prediction:.4f}")
        return jsonify({
            'patient_id': patient_id,
            'prediction': float(prediction),
            'confidence': float(confidence),
            'ecg_signal': ecg_signal.tolist(),
            'ecg_time': ecg_time.tolist(),
            'abnormalities': abnormalities,
            'shap_values': shap_values
        })

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def save_patient_data(patient_data, ecg_signal, ecg_time, abnormalities, prediction, confidence, shap_values):
    """
    Save patient data to JSON file
    """
    # Generate unique patient ID
    patient_id = f"patient_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Create data object
    data = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'patient_data': patient_data,
        'ecg_signal': ecg_signal.tolist(),
        'ecg_time': ecg_time.tolist(),
        'abnormalities': abnormalities,
        'prediction': float(prediction),
        'confidence': float(confidence),
        'shap_values': shap_values
    }

    # Save to file
    with open(f'data/patients/{patient_id}.json', 'w') as f:
        json.dump(data, f, indent=2)

    return patient_id

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """
    Endpoint to get all patient records
    """
    try:
        patients = []
        for filename in os.listdir('data/patients'):
            if filename.endswith('.json'):
                with open(f'data/patients/{filename}', 'r') as f:
                    patient = json.load(f)
                    # Add only summary data to avoid large response
                    patients.append({
                        'patient_id': patient['patient_id'],
                        'timestamp': patient['timestamp'],
                        'name': patient['patient_data'].get('name', 'Unknown'),
                        'age': patient['patient_data'].get('age', 0),
                        'gender': patient['patient_data'].get('gender', 'Unknown'),
                        'prediction': patient['prediction'],
                        'confidence': patient['confidence']
                    })

        return jsonify(patients)

    except Exception as e:
        print(f"Error getting patients: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """
    Endpoint to get a specific patient record
    """
    try:
        with open(f'data/patients/{patient_id}.json', 'r') as f:
            patient = json.load(f)

        return jsonify(patient)

    except Exception as e:
        print(f"Error getting patient {patient_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting simplified Flask server on port 8000...")
    app.run(debug=False, port=8000, host='0.0.0.0')
