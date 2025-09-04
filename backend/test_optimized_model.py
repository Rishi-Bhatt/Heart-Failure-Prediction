"""
Test Optimized Model

This script tests the optimized Random Forest model with the new ensemble weights
to verify improved accuracy for low, medium, and high risk patients.
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

def test_optimized_model():
    """
    Test the optimized model with different risk profiles
    """
    print("\nTesting Optimized Model with Different Risk Profiles")
    print("=" * 60)
    
    # Create a heart failure model instance
    model = HeartFailureModel()
    
    # Test patients with different risk profiles
    test_patients = [
        {
            "name": "Low Risk Patient",
            "age": 30,
            "gender": "Female",
            "blood_pressure": "110/70",
            "cholesterol": 150,
            "fasting_blood_sugar": 90,
            "resting_ecg": "Normal",
            "max_heart_rate": 180,
            "exercise_induced_angina": False,
            "st_depression": 0.0,
            "st_slope": "Flat",
            "num_major_vessels": 0,
            "thalassemia": "Normal",
            "prior_cardiac_event": {
                "type": "None",
                "time_since_event": 0,
                "severity": "None"
            },
            "biomarkers": {
                "nt_probnp": 50,
                "troponin": 0.01,
                "crp": 1.0
            },
            "weight": 60,
            "height": 165,
            "hdl": 60
        },
        {
            "name": "Medium Risk Patient",
            "age": 55,
            "gender": "Male",
            "blood_pressure": "140/90",
            "cholesterol": 210,
            "fasting_blood_sugar": 110,
            "resting_ecg": "Normal",
            "max_heart_rate": 150,
            "exercise_induced_angina": False,
            "st_depression": 1.0,
            "st_slope": "Flat",
            "num_major_vessels": 0,
            "thalassemia": "Normal",
            "prior_cardiac_event": {
                "type": "None",
                "time_since_event": 0,
                "severity": "None"
            },
            "biomarkers": {
                "nt_probnp": 150,
                "troponin": 0.02,
                "crp": 2.0
            },
            "weight": 80,
            "height": 175,
            "hdl": 45
        },
        {
            "name": "High Risk Patient",
            "age": 75,
            "gender": "Male",
            "blood_pressure": "180/100",
            "cholesterol": 280,
            "fasting_blood_sugar": 130,
            "resting_ecg": "Abnormal",
            "max_heart_rate": 110,
            "exercise_induced_angina": True,
            "st_depression": 2.5,
            "st_slope": "Flat",
            "num_major_vessels": 2,
            "thalassemia": "Reversible Defect",
            "prior_cardiac_event": {
                "type": "Myocardial Infarction",
                "time_since_event": 12,
                "severity": "Moderate"
            },
            "biomarkers": {
                "nt_probnp": 500,
                "troponin": 0.05,
                "crp": 5.0
            },
            "weight": 90,
            "height": 170,
            "hdl": 35
        }
    ]
    
    # Test each patient
    for patient in test_patients:
        print(f"\nPatient: {patient['name']}")
        print("-" * 40)
        
        # Generate ECG and analyze for abnormalities
        ecg_signal, ecg_time = generate_ecg(patient)
        abnormalities = analyze_ecg(ecg_signal, ecg_time, patient)
        
        # Preprocess data
        try:
            features = model.preprocess_data(patient, abnormalities)
            
            # Make prediction
            prediction, confidence, shap_values = model.predict(features, debug=True)
            
            # Get risk category
            if prediction < 0.12:
                risk_category = "Low"
            elif prediction < 0.28:
                risk_category = "Medium"
            else:
                risk_category = "High"
            
            print(f"\nPrediction: {prediction:.4f}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Risk Category: {risk_category}")
            
            # Print top 5 features by importance
            if shap_values and 'values' in shap_values and 'feature_names' in shap_values:
                feature_importance = list(zip(shap_values['feature_names'], shap_values['values']))
                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                print("\nTop 5 Features by Importance:")
                for feature, importance in feature_importance[:5]:
                    print(f"  {feature}: {importance:.4f}")
            
        except Exception as e:
            print(f"Error during test: {str(e)}")
    
    print("\nOptimized model test completed!")

if __name__ == "__main__":
    test_optimized_model()
