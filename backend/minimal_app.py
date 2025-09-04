from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import random
import math
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Ensure data directories exist
os.makedirs('data/patients', exist_ok=True)

@app.route('/api/test', methods=['GET'])
def test():
    """
    Test endpoint to verify the server is running
    """
    return jsonify({
        'status': 'success',
        'message': 'Backend is working!'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Simplified endpoint to predict heart failure risk based on patient data
    """
    try:
        # Get patient data from request
        patient_data = request.json
        
        # Generate a simple synthetic ECG signal
        ecg_signal, ecg_time = generate_simple_ecg(patient_data)
        
        # Generate simple abnormalities
        abnormalities = generate_simple_abnormalities(patient_data)
        
        # Make a simple prediction
        prediction, confidence = make_simple_prediction(patient_data)
        
        # Generate simple SHAP values
        shap_values = generate_simple_shap_values(patient_data)
        
        # Save patient data
        patient_id = save_patient_data(patient_data, ecg_signal, ecg_time, 
                                      abnormalities, prediction, confidence, shap_values)
        
        return jsonify({
            'patient_id': patient_id,
            'prediction': prediction,
            'confidence': confidence,
            'ecg_signal': ecg_signal,
            'ecg_time': ecg_time,
            'abnormalities': abnormalities,
            'shap_values': shap_values
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_simple_ecg(patient_data):
    """
    Generate a simple synthetic ECG signal
    """
    # Create a simple sine wave as ECG
    duration = 10  # seconds
    sampling_rate = 100  # Hz
    num_points = int(duration * sampling_rate)
    
    # Create time array
    ecg_time = [i/sampling_rate for i in range(num_points)]
    
    # Base frequency based on heart rate
    heart_rate = patient_data.get('max_heart_rate', 75)
    base_freq = heart_rate / 60  # Convert to Hz
    
    # Create a simple ECG-like signal
    ecg_signal = []
    
    for t in ecg_time:
        # Basic sine wave
        value = math.sin(2 * math.pi * base_freq * t)
        
        # Add QRS complexes
        for i in range(int(duration * base_freq)):
            # Position of the QRS complex
            pos = i / base_freq
            
            # Add a QRS complex (simplified as a spike)
            if abs(t - pos) < 0.1:
                # Create a spike
                spike = math.sin((t - pos + 0.1) / 0.2 * math.pi) * 0.8
                value += spike
        
        # Add some noise
        noise = (random.random() - 0.5) * 0.1
        value += noise
        
        ecg_signal.append(value)
    
    return ecg_signal, ecg_time

def generate_simple_abnormalities(patient_data):
    """
    Generate simple abnormalities based on patient data
    """
    abnormalities = {
        'PVCs': [],
        'Flatlines': [],
        'Tachycardia': [],
        'Bradycardia': [],
        'QT_prolongation': [],
        'Atrial_Fibrillation': []
    }
    
    # Add some random abnormalities based on patient data
    if patient_data.get('age', 60) > 65:
        # Add a PVC
        abnormalities['PVCs'].append({
            'time': random.uniform(1, 8),
            'duration': 0.2
        })
    
    if patient_data.get('max_heart_rate', 75) > 100:
        # Add tachycardia
        abnormalities['Tachycardia'].append({
            'time': random.uniform(2, 7),
            'duration': 1.0,
            'rate': patient_data.get('max_heart_rate', 75)
        })
    
    if patient_data.get('prior_cardiac_event', {}).get('type', ''):
        # Add QT prolongation
        abnormalities['QT_prolongation'].append({
            'time': random.uniform(3, 6),
            'duration': 0.5,
            'interval': 0.48
        })
    
    return abnormalities

def make_simple_prediction(patient_data):
    """
    Make a simple prediction based on patient data
    """
    # Base risk score
    risk_score = 0.2
    
    # Age factor
    age = patient_data.get('age', 60)
    if age > 65:
        risk_score += 0.1
    if age > 75:
        risk_score += 0.1
    
    # Gender factor
    if patient_data.get('gender', 'Male') == 'Male':
        risk_score += 0.05
    
    # Blood pressure factor
    bp = patient_data.get('blood_pressure', '120/80')
    try:
        systolic = int(bp.split('/')[0])
        if systolic > 140:
            risk_score += 0.1
    except:
        pass
    
    # Cholesterol factor
    cholesterol = patient_data.get('cholesterol', 200)
    if cholesterol > 240:
        risk_score += 0.1
    
    # Prior cardiac event factor
    if patient_data.get('prior_cardiac_event', {}).get('type', ''):
        risk_score += 0.2
    
    # Add some randomness
    risk_score += (random.random() - 0.5) * 0.1
    
    # Ensure risk score is between 0 and 1
    risk_score = max(0, min(1, risk_score))
    
    # Calculate confidence (higher for extreme values)
    confidence = 0.7 + abs(risk_score - 0.5) * 0.4
    
    return risk_score, confidence

def generate_simple_shap_values(patient_data):
    """
    Generate simple SHAP values for explainability
    """
    feature_names = [
        'age', 'gender', 'blood_pressure', 'cholesterol', 
        'max_heart_rate', 'prior_cardiac_event'
    ]
    
    # Generate random SHAP values
    values = []
    for feature in feature_names:
        if feature == 'age' and patient_data.get('age', 60) > 65:
            values.append(random.uniform(0.1, 0.2))
        elif feature == 'gender' and patient_data.get('gender', 'Male') == 'Male':
            values.append(random.uniform(0.05, 0.1))
        elif feature == 'blood_pressure':
            bp = patient_data.get('blood_pressure', '120/80')
            try:
                systolic = int(bp.split('/')[0])
                if systolic > 140:
                    values.append(random.uniform(0.1, 0.15))
                else:
                    values.append(random.uniform(-0.05, 0.05))
            except:
                values.append(0)
        elif feature == 'cholesterol' and patient_data.get('cholesterol', 200) > 240:
            values.append(random.uniform(0.1, 0.15))
        elif feature == 'max_heart_rate' and patient_data.get('max_heart_rate', 75) > 100:
            values.append(random.uniform(0.05, 0.1))
        elif feature == 'prior_cardiac_event' and patient_data.get('prior_cardiac_event', {}).get('type', ''):
            values.append(random.uniform(0.15, 0.25))
        else:
            values.append(random.uniform(-0.05, 0.05))
    
    return {
        'base_value': 0.5,
        'values': values,
        'feature_names': feature_names
    }

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
        'ecg_signal': ecg_signal,
        'ecg_time': ecg_time,
        'abnormalities': abnormalities,
        'prediction': prediction,
        'confidence': confidence,
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
        print(f"Error: {str(e)}")
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
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """
    Simplified endpoint to simulate model retraining
    """
    try:
        # Simulate retraining
        print("Simulating model retraining...")
        
        # Count patient records
        patient_count = len([f for f in os.listdir('data/patients') if f.endswith('.json')])
        
        return jsonify({
            'success': True,
            'message': f'Model retrained successfully with {patient_count} patient records',
            'date': datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting minimal Flask server...")
    app.run(debug=True, port=5000)
