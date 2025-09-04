"""
Test Risk Categories

This script tests the risk categories for different patient profiles.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from models.heart_failure_model import HeartFailureModel
from utils.ecg_generator import generate_ecg, analyze_ecg
from risk_calibration import get_risk_category, get_risk_score_explanation

# Create test patients
test_patients = [
    {
        "name": "Low Risk Patient",
        "age": 30,
        "gender": "Female",
        "blood_pressure": "110/70",
        "cholesterol": 150,
        "fasting_blood_sugar": 85,
        "resting_ecg": "Normal",
        "max_heart_rate": 180,
        "exercise_induced_angina": False,
        "st_depression": 0.0,
        "st_slope": "Upsloping",
        "num_major_vessels": 0,
        "thalassemia": "Normal",
        "prior_cardiac_event": {"type": "None", "time_since_event": 0, "severity": "None", "location": "None"},
        "biomarkers": {"nt_probnp": 50, "troponin": 0.005, "crp": 0.5}
    },
    {
        "name": "Medium Risk Patient",
        "age": 55,
        "gender": "Male",
        "blood_pressure": "140/85",
        "cholesterol": 210,
        "fasting_blood_sugar": 110,
        "resting_ecg": "Normal",
        "max_heart_rate": 150,
        "exercise_induced_angina": False,
        "st_depression": 1.0,
        "st_slope": "Flat",
        "num_major_vessels": 1,
        "thalassemia": "Normal",
        "prior_cardiac_event": {"type": "None", "time_since_event": 0, "severity": "None", "location": "None"},
        "biomarkers": {"nt_probnp": 200, "troponin": 0.02, "crp": 3}
    },
    {
        "name": "High Risk Patient",
        "age": 75,
        "gender": "Male",
        "blood_pressure": "180/100",
        "cholesterol": 280,
        "fasting_blood_sugar": 180,
        "resting_ecg": "Left Ventricular Hypertrophy",
        "max_heart_rate": 110,
        "exercise_induced_angina": True,
        "st_depression": 2.5,
        "st_slope": "Downsloping",
        "num_major_vessels": 3,
        "thalassemia": "Reversible Defect",
        "prior_cardiac_event": {"type": "Myocardial Infarction", "time_since_event": 12, "severity": "Moderate", "location": "Anterior"},
        "biomarkers": {"nt_probnp": 1000, "troponin": 0.1, "crp": 10}
    },
    {
        "name": "Very High Risk Patient",
        "age": 85,
        "gender": "Male",
        "blood_pressure": "200/110",
        "cholesterol": 350,
        "fasting_blood_sugar": 250,
        "resting_ecg": "Left Ventricular Hypertrophy",
        "max_heart_rate": 90,
        "exercise_induced_angina": True,
        "st_depression": 4.0,
        "st_slope": "Downsloping",
        "num_major_vessels": 5,
        "thalassemia": "Reversible Defect",
        "prior_cardiac_event": {"type": "Myocardial Infarction", "time_since_event": 1, "severity": "Severe", "location": "Anterior"},
        "biomarkers": {"nt_probnp": 5000, "troponin": 0.5, "crp": 30}
    }
]

# Create a heart failure model instance
model = HeartFailureModel()

# Test each patient
print("\nTesting Risk Categories for Different Patient Profiles")
print("=" * 60)

for patient in test_patients:
    print(f"\nPatient: {patient['name']}")
    print("-" * 40)
    
    # Generate ECG and analyze for abnormalities
    ecg_signal, ecg_time = generate_ecg(patient)
    abnormalities = analyze_ecg(ecg_signal, ecg_time, patient)
    
    # Preprocess data
    features = model.preprocess_data(patient, abnormalities)
    
    # Make prediction with debug mode
    prediction, confidence, shap_values = model.predict(features, debug=True)
    
    # Get risk explanation
    risk_explanation = get_risk_score_explanation(prediction, patient)
    
    print(f"\nPrediction: {prediction:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Risk Category: {risk_explanation['risk_category']}")
    print(f"\nRisk Thresholds:")
    print(f"Base: Low-Medium = {risk_explanation['thresholds']['base']['low_medium']:.4f}, Medium-High = {risk_explanation['thresholds']['base']['medium_high']:.4f}")
    print(f"Age/Gender Adjusted: Low-Medium = {risk_explanation['thresholds']['age_gender_adjusted']['low_medium']:.4f}, Medium-High = {risk_explanation['thresholds']['age_gender_adjusted']['medium_high']:.4f}")
    print(f"Final: Low-Medium = {risk_explanation['thresholds']['final']['low_medium']:.4f}, Medium-High = {risk_explanation['thresholds']['final']['medium_high']:.4f}")
    
    print("\n" + "=" * 60)
