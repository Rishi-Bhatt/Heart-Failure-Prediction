"""
Test Feature Scaling Fix

This script tests if the feature scaling fix resolves the issue with
the 'Feature names unseen at fit time' error.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import heart failure model
from models.heart_failure_model import HeartFailureModel
from utils.ecg_generator import generate_ecg, analyze_ecg

def test_feature_scaling_fix():
    """
    Test if the feature scaling fix resolves the issue
    """
    print("Testing feature scaling fix...")
    
    # Create a test patient with all required features
    test_patient = {
        "name": "Test Patient",
        "age": 65,
        "gender": "Male",
        "blood_pressure": "140/90",
        "cholesterol": 220,
        "fasting_blood_sugar": 110,
        "resting_ecg": "Normal",
        "max_heart_rate": 140,
        "exercise_induced_angina": False,
        "st_depression": 0.5,
        "st_slope": "Flat",
        "num_major_vessels": 1,
        "thalassemia": "Normal",
        "prior_cardiac_event": {
            "type": "None",
            "time_since_event": 0,
            "severity": "None",
            "location": "None"
        },
        "biomarkers": {
            "nt_probnp": 100,
            "troponin": 0.01,
            "crp": 1.5
        },
        "weight": 75,
        "height": 175,
        "hdl": 50
    }
    
    # Create a heart failure model instance
    model = HeartFailureModel()
    
    # Generate ECG and analyze for abnormalities
    ecg_signal, ecg_time = generate_ecg(test_patient)
    abnormalities = analyze_ecg(ecg_signal, ecg_time, test_patient)
    
    # Preprocess data
    try:
        features = model.preprocess_data(test_patient, abnormalities)
        print("Preprocessing successful!")
        
        # Make prediction
        prediction, confidence, shap_values = model.predict(features, debug=True)
        print(f"\nPrediction: {prediction:.4f}")
        print(f"Confidence: {confidence:.4f}")
        
        print("\nFeature scaling fix test completed successfully!")
        return True
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    test_feature_scaling_fix()
