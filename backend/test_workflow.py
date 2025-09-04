import os
import sys
import json
import numpy as np
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting workflow test...")

try:
    # Import simplified modules
    print("Importing modules...")
    from utils.simple_ecg_generator import generate_simple_ecg, analyze_simple_ecg
    from models.simple_model import SimpleHeartFailureModel
    
    # Initialize model
    print("Initializing model...")
    model = SimpleHeartFailureModel()
    
    # Create sample patient data
    print("Creating sample patient data...")
    patient_data = {
        'name': 'Test Patient',
        'age': 65,
        'gender': 'Male',
        'blood_pressure': '140/90',
        'cholesterol': 220,
        'fasting_blood_sugar': 110,
        'chest_pain_type': 'Typical Angina',
        'ecg_result': 'Normal',
        'max_heart_rate': 140,
        'exercise_induced_angina': True,
        'st_depression': 1.5,
        'slope_of_st': 'Flat',
        'number_of_major_vessels': 2,
        'thalassemia': 'Normal',
        'prior_cardiac_event': {
            'type': 'Myocardial Infarction',
            'time_since_event': 6,
            'severity': 'Moderate'
        },
        'medications': [
            {
                'type': 'Beta-blockers',
                'time_of_administration': 2
            },
            {
                'type': 'ACE inhibitors',
                'time_of_administration': 4
            }
        ]
    }
    
    # Generate synthetic ECG
    print("Generating ECG signal...")
    ecg_signal, ecg_time = generate_simple_ecg(patient_data)
    print(f"ECG signal shape: {ecg_signal.shape}")
    
    # Analyze ECG
    print("Analyzing ECG signal...")
    abnormalities = analyze_simple_ecg(ecg_signal, ecg_time, patient_data)
    print(f"Detected abnormalities: {abnormalities}")
    
    # Make prediction
    print("Making prediction...")
    features = np.array([[
        patient_data.get('age', 60),
        1 if patient_data.get('gender', 'Male') == 'Male' else 0,
        patient_data.get('cholesterol', 200),
        patient_data.get('max_heart_rate', 75)
    ]])
    prediction, confidence, shap_values = model.predict(features)
    print(f"Prediction: {prediction:.4f}")
    print(f"Confidence: {confidence:.4f}")
    
    # Save patient data
    print("Saving patient data...")
    os.makedirs('data/patients', exist_ok=True)
    patient_id = f"patient_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
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
    
    with open(f'data/patients/{patient_id}.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Patient data saved with ID: {patient_id}")
    
    # Read patient data back
    print("Reading patient data...")
    with open(f'data/patients/{patient_id}.json', 'r') as f:
        loaded_data = json.load(f)
    
    print(f"Loaded patient ID: {loaded_data['patient_id']}")
    print(f"Loaded prediction: {loaded_data['prediction']:.4f}")
    
    print("Workflow test completed successfully!")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("Test completed.")
