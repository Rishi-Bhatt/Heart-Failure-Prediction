import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Create a simple model class for testing
class HeartFailureModel:
    def __init__(self):
        self.model_path = 'data/model.joblib'
        self.scaler_path = 'data/scaler.joblib'
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
            'exercise_induced_angina', 'st_depression', 'st_slope',
            'num_major_vessels', 'thalassemia', 'prior_event_severity',
            'time_since_event', 'pvc_count', 'qt_prolongation',
            'af_detected', 'tachycardia_detected', 'bradycardia_detected'
        ]

    def _encode_chest_pain(self, chest_pain_type):
        mapping = {
            'Typical Angina': 0,
            'Atypical Angina': 1,
            'Non-Anginal Pain': 2,
            'Asymptomatic': 3
        }
        return mapping.get(chest_pain_type, 0)

    def _encode_resting_ecg(self, ecg_result):
        mapping = {
            'Normal': 0,
            'ST-T Wave Abnormality': 1,
            'Left Ventricular Hypertrophy': 2
        }
        return mapping.get(ecg_result, 0)

    def _encode_st_slope(self, st_slope):
        mapping = {
            'Upsloping': 0,
            'Flat': 1,
            'Downsloping': 2
        }
        return mapping.get(st_slope, 0)

    def _encode_thalassemia(self, thalassemia):
        mapping = {
            'Normal': 0,
            'Fixed Defect': 1,
            'Reversible Defect': 2
        }
        return mapping.get(thalassemia, 0)

# Create a test patient
test_patient = {
    'patient_id': 'test_patient',
    'timestamp': datetime.now().isoformat(),
    'patient_data': {
        'name': 'Test Patient',
        'age': 60,
        'gender': 'Male',
        'blood_pressure': '120/80',
        'cholesterol': 200,
        'fasting_blood_sugar': 110,
        'chest_pain_type': 'Typical Angina',
        'ecg_result': 'Normal',
        'max_heart_rate': 150,
        'exercise_induced_angina': False,
        'st_depression': 0.0,
        'slope_of_st': 'Flat',
        'number_of_major_vessels': 0,
        'thalassemia': 'Normal'
    },
    'prediction': 0.2,
    'confidence': 0.8,
    'abnormalities': {
        'PVCs': [],
        'QT_prolongation': [],
        'Atrial_Fibrillation': [],
        'Tachycardia': [],
        'Bradycardia': []
    }
}

# Save test patient
os.makedirs('data/patients', exist_ok=True)
with open('data/patients/test_patient.json', 'w') as f:
    json.dump(test_patient, f, indent=2)

# Import the model retrainer
from retraining.model_retrainer import ModelRetrainer

# Create model and retrainer
model = HeartFailureModel()
retrainer = ModelRetrainer(model)

# Test retraining
print("Testing model retraining...")
result = retrainer.retrain()
print(f"Retraining result: {result}")

# Clean up
os.remove('data/patients/test_patient.json')
print("Test complete.")
