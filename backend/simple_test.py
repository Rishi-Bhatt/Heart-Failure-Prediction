"""
Simple Test for Heart Failure Model
"""

import sys
import json
from datetime import datetime

# Import the model
from models.heart_failure_model import HeartFailureModel

def main():
    """
    Main function to run tests
    """
    print("Simple Test for Heart Failure Model")
    print("=" * 50)
    
    # Initialize model
    model = HeartFailureModel()
    
    # Test low risk patient
    print("\nTesting Low Risk Patient")
    print("-" * 30)
    
    low_risk_patient = {
        "name": "John Healthy",
        "age": 45,
        "gender": "Male",
        "blood_pressure": "120/80",
        "cholesterol": 180,
        "fasting_blood_sugar": 90,
        "resting_ecg": "Normal",
        "max_heart_rate": 170,
        "exercise_induced_angina": False,
        "st_depression": 0.2,
        "st_slope": "Upsloping",
        "num_major_vessels": 0,
        "thalassemia": "Normal",
        "biomarkers": {
            "nt_probnp": 50,
            "troponin": 0.01,
            "crp": 1.5
        }
    }
    
    # Create mock abnormalities
    abnormalities = {
        'PVCs': [],
        'Flatlines': [],
        'Tachycardia': [],
        'Bradycardia': [],
        'QT_prolongation': [],
        'Atrial_Fibrillation': []
    }
    
    # Preprocess data
    features = model.preprocess_data(low_risk_patient, abnormalities)
    
    # Make prediction with debug output
    prediction, confidence, shap_values = model.predict(features, debug=True)
    
    # Determine risk category
    if prediction < 0.15:
        risk_category = 'Low'
    elif prediction < 0.35:
        risk_category = 'Medium'
    else:
        risk_category = 'High'
    
    # Print results
    print(f"\nPrediction: {prediction:.4f}")
    print(f"Risk Category: {risk_category}")
    print(f"Expected Risk: Low")
    print(f"Matches Expected: {'Yes' if risk_category == 'Low' else 'No'}")
    
    # Test high risk patient
    print("\nTesting High Risk Patient")
    print("-" * 30)
    
    high_risk_patient = {
        "name": "Robert Critical",
        "age": 72,
        "gender": "Male",
        "blood_pressure": "160/95",
        "cholesterol": 260,
        "fasting_blood_sugar": 140,
        "resting_ecg": "Left Ventricular Hypertrophy",
        "max_heart_rate": 120,
        "exercise_induced_angina": True,
        "st_depression": 2.5,
        "st_slope": "Downsloping",
        "num_major_vessels": 3,
        "thalassemia": "Reversible Defect",
        "prior_cardiac_event": {
            "type": "Myocardial Infarction",
            "time_since_event": 6,
            "severity": "Severe",
            "location": "Anterior"
        },
        "biomarkers": {
            "nt_probnp": 450,
            "troponin": 0.09,
            "crp": 8.5
        }
    }
    
    # Preprocess data
    features = model.preprocess_data(high_risk_patient, abnormalities)
    
    # Make prediction with debug output
    prediction, confidence, shap_values = model.predict(features, debug=True)
    
    # Determine risk category
    if prediction < 0.15:
        risk_category = 'Low'
    elif prediction < 0.35:
        risk_category = 'Medium'
    else:
        risk_category = 'High'
    
    # Print results
    print(f"\nPrediction: {prediction:.4f}")
    print(f"Risk Category: {risk_category}")
    print(f"Expected Risk: High")
    print(f"Matches Expected: {'Yes' if risk_category == 'High' else 'No'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
