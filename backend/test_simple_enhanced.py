"""
Test Simple Enhanced Model

This script tests the enhanced heart failure model with a simpler approach.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_enhanced():
    """
    Test the enhanced heart failure model with a simpler approach
    """
    print("\nTesting Enhanced Heart Failure Model (Simple)")
    print("=" * 60)

    # Load model and scaler
    model_path = 'models/heart_failure_model.joblib'
    scaler_path = 'models/heart_failure_scaler.joblib'

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Loaded model and scaler successfully")
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        return

    # Create test data for different risk profiles
    test_data = [
        # Low risk patient
        {
            "name": "Low Risk Patient",
            "age": 30,
            "sex": 0,  # Female
            "chest_pain_type": 0,
            "resting_bp": 110,
            "cholesterol": 150,
            "fasting_blood_sugar": 0,
            "resting_ecg": 0,
            "max_heart_rate": 180,
            "exercise_induced_angina": 0,
            "st_depression": 0.0,
            "st_slope": 1,
            "num_major_vessels": 0,
            "thalassemia": 0,
            "prior_event_severity": 0,
            "time_since_event": 0,
            "pvc_count": 0,
            "qt_prolongation": 0,
            "af_detected": 0,
            "tachycardia_detected": 0,
            "bradycardia_detected": 0,
            "age_squared": 900,
            "bmi": 22.0,
            "bp_age_ratio": 3.67,
            "cholesterol_hdl_ratio": 2.5,
            "heart_rate_recovery": 6.0,
            "nt_probnp": 50,
            "troponin": 0.01,
            "crp": 1.0,
            "bnp": 50,
            "creatinine": 0.8,
            "ace_inhibitor": 0,
            "arb": 0,
            "beta_blocker": 0,
            "statin": 0,
            "antiplatelet": 0,
            "diuretic": 0,
            "calcium_channel_blocker": 0
        },
        # Medium risk patient with medications
        {
            "name": "Medium Risk Patient with Medications",
            "age": 55,
            "sex": 1,  # Male
            "chest_pain_type": 1,
            "resting_bp": 140,
            "cholesterol": 210,
            "fasting_blood_sugar": 0,
            "resting_ecg": 0,
            "max_heart_rate": 150,
            "exercise_induced_angina": 0,
            "st_depression": 1.0,
            "st_slope": 1,
            "num_major_vessels": 0,
            "thalassemia": 1,
            "prior_event_severity": 1,
            "time_since_event": 24,
            "pvc_count": 0,
            "qt_prolongation": 0,
            "af_detected": 0,
            "tachycardia_detected": 0,
            "bradycardia_detected": 0,
            "age_squared": 3025,
            "bmi": 26.1,
            "bp_age_ratio": 2.55,
            "cholesterol_hdl_ratio": 4.67,
            "heart_rate_recovery": 3.0,
            "nt_probnp": 150,
            "troponin": 0.02,
            "crp": 2.0,
            "bnp": 120,
            "creatinine": 1.1,
            "ace_inhibitor": 1,
            "arb": 0,
            "beta_blocker": 1,
            "statin": 1,
            "antiplatelet": 0,
            "diuretic": 0,
            "calcium_channel_blocker": 0
        },
        # High risk patient with elevated biomarkers
        {
            "name": "High Risk Patient with Elevated Biomarkers",
            "age": 75,
            "sex": 1,  # Male
            "chest_pain_type": 3,
            "resting_bp": 180,
            "cholesterol": 280,
            "fasting_blood_sugar": 1,
            "resting_ecg": 1,
            "max_heart_rate": 110,
            "exercise_induced_angina": 1,
            "st_depression": 2.5,
            "st_slope": 2,
            "num_major_vessels": 2,
            "thalassemia": 2,
            "prior_event_severity": 3,
            "time_since_event": 6,
            "pvc_count": 0,
            "qt_prolongation": 1,
            "af_detected": 1,
            "tachycardia_detected": 0,
            "bradycardia_detected": 1,
            "age_squared": 5625,
            "bmi": 31.1,
            "bp_age_ratio": 2.4,
            "cholesterol_hdl_ratio": 8.0,
            "heart_rate_recovery": 0.0,
            "nt_probnp": 1200,
            "troponin": 0.05,
            "crp": 12.0,
            "bnp": 450,
            "creatinine": 1.6,
            "ace_inhibitor": 1,
            "arb": 0,
            "beta_blocker": 1,
            "statin": 1,
            "antiplatelet": 1,
            "diuretic": 1,
            "calcium_channel_blocker": 0
        }
    ]

    # Convert test data to DataFrame
    df = pd.DataFrame(test_data)

    # Get only the features that the model was trained on
    model_features = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
        'exercise_induced_angina', 'st_depression', 'st_slope',
        'num_major_vessels', 'thalassemia', 'prior_event_severity',
        'time_since_event', 'pvc_count', 'qt_prolongation',
        'af_detected', 'tachycardia_detected', 'bradycardia_detected',
        'age_squared', 'bmi', 'bp_age_ratio', 'cholesterol_hdl_ratio', 'heart_rate_recovery'
    ]

    # Get all feature names for rule-based and logistic regression
    all_feature_names = [col for col in df.columns if col != 'name']

    # Test each patient
    for i, patient in enumerate(test_data):
        print(f"\nPatient: {patient['name']}")
        print("-" * 40)

        # Extract features for this patient
        patient_features = df.iloc[i][model_features].values.reshape(1, -1)

        # Scale features
        patient_features_scaled = scaler.transform(patient_features)

        # Get Random Forest prediction
        rf_prediction = model.predict_proba(patient_features_scaled)[0, 1]

        # Create DataFrame for rule-based and logistic regression predictions
        patient_df = pd.DataFrame([{col: df.iloc[i][col] for col in all_feature_names}])

        # Import rule-based and logistic regression functions
        from models.heart_failure_model import HeartFailureModel
        hf_model = HeartFailureModel()

        # Get rule-based prediction
        rule_prediction = hf_model._rule_based_prediction(patient_df, debug=True)

        # Get logistic regression prediction
        lr_prediction = hf_model._simplified_lr_prediction(patient_df, debug=True)

        # Calculate ensemble prediction
        ensemble_prediction = 0.15 * rule_prediction + 0.15 * lr_prediction + 0.70 * rf_prediction

        # Get risk category
        if ensemble_prediction < 0.12:
            risk_category = "Low"
        elif ensemble_prediction < 0.28:
            risk_category = "Medium"
        else:
            risk_category = "High"

        print(f"\nRandom Forest Prediction: {rf_prediction:.4f}")
        print(f"Rule-Based Prediction: {rule_prediction:.4f}")
        print(f"Logistic Regression Prediction: {lr_prediction:.4f}")
        print(f"Ensemble Prediction: {ensemble_prediction:.4f}")
        print(f"Risk Category: {risk_category}")

    print("\nEnhanced model test completed!")

if __name__ == "__main__":
    test_simple_enhanced()
