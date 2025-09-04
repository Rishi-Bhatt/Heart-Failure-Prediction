"""
Test Prediction with Gradient Boosting

This script tests if gradient boosting is being used in the prediction.
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

# Create a test patient
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
    "debug_mode": True
}

# Create a heart failure model instance
model = HeartFailureModel()

# Generate ECG and analyze for abnormalities
ecg_signal, ecg_time = generate_ecg(test_patient)
abnormalities = analyze_ecg(ecg_signal, ecg_time, test_patient)

# Preprocess data
features = model.preprocess_data(test_patient, abnormalities)

# Make prediction with debug mode
prediction, confidence, shap_values = model.predict(features, debug=True)

# Get risk explanation
from risk_calibration import get_risk_score_explanation
risk_explanation = get_risk_score_explanation(prediction, test_patient)

print(f"\nPrediction: {prediction:.4f}")
print(f"Confidence: {confidence:.4f}")
print(f"Risk Category: {risk_explanation['risk_category']}")
print(f"\nRisk Thresholds:")
print(f"Base: Low-Medium = {risk_explanation['thresholds']['base']['low_medium']:.4f}, Medium-High = {risk_explanation['thresholds']['base']['medium_high']:.4f}")
print(f"Age/Gender Adjusted: Low-Medium = {risk_explanation['thresholds']['age_gender_adjusted']['low_medium']:.4f}, Medium-High = {risk_explanation['thresholds']['age_gender_adjusted']['medium_high']:.4f}")
print(f"Final: Low-Medium = {risk_explanation['thresholds']['final']['low_medium']:.4f}, Medium-High = {risk_explanation['thresholds']['final']['medium_high']:.4f}")
