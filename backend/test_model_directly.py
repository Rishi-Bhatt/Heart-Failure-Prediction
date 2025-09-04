"""
Test Heart Failure Model Directly

This script tests the heart failure model directly without going through the API.
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Import the model
from models.heart_failure_model import HeartFailureModel

# Test cases with different risk profiles
TEST_CASES = [
    {
        "name": "Low Risk Patient",
        "patient_data": {
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
        },
        "expected_risk": "Low"
    },
    {
        "name": "Medium Risk Patient",
        "patient_data": {
            "name": "Mary Moderate",
            "age": 58,
            "gender": "Female",
            "blood_pressure": "140/90",
            "cholesterol": 220,
            "fasting_blood_sugar": 110,
            "resting_ecg": "ST-T Wave Abnormality",
            "max_heart_rate": 150,
            "exercise_induced_angina": True,
            "st_depression": 1.0,
            "st_slope": "Flat",
            "num_major_vessels": 1,
            "thalassemia": "Fixed Defect",
            "prior_cardiac_event": {
                "type": "Angina",
                "time_since_event": 24,
                "severity": "Mild",
                "location": "Lateral"
            },
            "biomarkers": {
                "nt_probnp": 150,
                "troponin": 0.03,
                "crp": 3.5
            }
        },
        "expected_risk": "Medium"
    },
    {
        "name": "High Risk Patient",
        "patient_data": {
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
        },
        "expected_risk": "High"
    }
]

def main():
    """
    Main function to run tests
    """
    print("Testing Heart Failure Model Directly")
    print("=" * 50)

    # Initialize model
    model = HeartFailureModel()

    results = []

    for test_case in TEST_CASES:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 30)

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
        features = model.preprocess_data(test_case['patient_data'], abnormalities)

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
        print(f"Expected Risk: {test_case['expected_risk']}")
        print(f"Matches Expected: {'Yes' if risk_category == test_case['expected_risk'] else 'No'}")

        # Store results
        results.append({
            "name": test_case['name'],
            "prediction": prediction,
            "risk_category": risk_category,
            "expected_risk": test_case['expected_risk'],
            "matches_expected": risk_category == test_case['expected_risk']
        })

    # Print summary
    print("\nSummary")
    print("=" * 50)

    success_count = sum(1 for r in results if r['matches_expected'])
    total_count = len(results)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0

    print(f"Total Tests: {total_count}")
    print(f"Successful Tests: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")

    # Save results to file
    results_file = f"direct_model_test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_count,
            "successful_tests": success_count,
            "success_rate": success_rate,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
