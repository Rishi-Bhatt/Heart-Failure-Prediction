"""
Test Random Forest Model

This script tests the Random Forest model directly to ensure it works correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rf_model():
    """
    Test the Random Forest model directly
    """
    print("\nTesting Random Forest Model Directly")
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
            "heart_rate_recovery": 6.0
        },
        # Medium risk patient
        {
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
            "thalassemia": 0,
            "prior_event_severity": 0,
            "time_since_event": 0,
            "pvc_count": 0,
            "qt_prolongation": 0,
            "af_detected": 0,
            "tachycardia_detected": 0,
            "bradycardia_detected": 0,
            "age_squared": 3025,
            "bmi": 26.1,
            "bp_age_ratio": 2.55,
            "cholesterol_hdl_ratio": 4.67,
            "heart_rate_recovery": 3.0
        },
        # High risk patient
        {
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
            "prior_event_severity": 2,
            "time_since_event": 12,
            "pvc_count": 0,
            "qt_prolongation": 1,
            "af_detected": 1,
            "tachycardia_detected": 0,
            "bradycardia_detected": 1,
            "age_squared": 5625,
            "bmi": 31.1,
            "bp_age_ratio": 2.4,
            "cholesterol_hdl_ratio": 8.0,
            "heart_rate_recovery": 0.0
        }
    ]
    
    # Convert test data to DataFrame
    df = pd.DataFrame(test_data)
    
    # Get feature names from model
    feature_names = df.columns.tolist()
    print(f"Number of features: {len(feature_names)}")
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Make predictions
    predictions = model.predict_proba(X_scaled)[:, 1]
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Print results
    for i, (patient, prediction) in enumerate(zip(test_data, predictions)):
        risk_level = "Low" if prediction < 0.12 else "Medium" if prediction < 0.28 else "High"
        print(f"\nPatient {i+1} (Risk: {risk_level}):")
        print(f"  Prediction: {prediction:.4f}")
        
        # Get top 5 features by importance
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print("  Top 5 Features by Importance:")
        for feature, importance in feature_importance[:5]:
            print(f"    {feature}: {importance:.4f}")
    
    print("\nRandom Forest model test completed!")

if __name__ == "__main__":
    test_rf_model()
