"""
Debug script for the heart failure model
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_model():
    """
    Debug the heart failure model
    """
    print("\nDebugging Heart Failure Model")
    print("=" * 60)
    
    # Load model and scaler
    model_path = 'models/heart_failure_model.joblib'
    scaler_path = 'models/heart_failure_scaler.joblib'
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Loaded model and scaler successfully")
        print(f"Model expects {model.n_features_in_} features")
        
        # Create test data
        test_data = {
            "age": 65,
            "sex": 1,
            "chest_pain_type": 1,
            "resting_bp": 140,
            "cholesterol": 220,
            "fasting_blood_sugar": 0,
            "resting_ecg": 0,
            "max_heart_rate": 150,
            "exercise_induced_angina": 0,
            "st_depression": 1.0,
            "st_slope": 1,
            "num_major_vessels": 0,
            "thalassemia": 1,
            "prior_event_severity": 1,
            "time_since_event": 12,
            "pvc_count": 0,
            "qt_prolongation": 0,
            "af_detected": 0,
            "tachycardia_detected": 0,
            "bradycardia_detected": 0,
            "age_squared": 4225,
            "bmi": 25.0,
            "bp_age_ratio": 2.15,
            "cholesterol_hdl_ratio": 4.4,
            "heart_rate_recovery": 3.0
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([test_data])
        
        # Get feature names from scaler
        feature_names = scaler.feature_names_in_
        print(f"Scaler expects these features: {feature_names}")
        
        # Ensure DataFrame has all required features in the correct order
        df_for_scaling = df[feature_names]
        
        # Scale features
        features_scaled = scaler.transform(df_for_scaling)
        
        # Make prediction
        prediction = model.predict_proba(features_scaled)[0, 1]
        
        print(f"\nPrediction: {prediction:.4f}")
        
        # Now try with the heart failure model class
        from models.heart_failure_model import HeartFailureModel
        hf_model = HeartFailureModel()
        
        # Create patient data
        patient_data = {
            "age": 65,
            "sex": 1,
            "chest_pain_type": 1,
            "resting_bp": "140/90",
            "cholesterol": 220,
            "fasting_blood_sugar": 0,
            "resting_ecg": 0,
            "max_heart_rate": 150,
            "exercise_induced_angina": 0,
            "st_depression": 1.0,
            "st_slope": 1,
            "num_major_vessels": 0,
            "thalassemia": 1,
            "biomarkers": {"nt_probnp": 300},
            "medications": [{"type": "Beta-blockers", "time_of_administration": 2}]
        }
        
        # Preprocess data
        features = hf_model.preprocess_data(patient_data, {})
        
        # Make prediction
        prediction, confidence, shap_values = hf_model.predict(features, debug=True)
        
        print(f"\nHeart Failure Model Prediction: {prediction:.4f}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model()
