"""
Test Risk Assessment Accuracy

This script tests the risk assessment accuracy by making predictions
for patients with different risk profiles.
"""

import sys
import json
import requests
from datetime import datetime
import time

# Test cases with different risk profiles
TEST_CASES = [
    {
        "name": "Low Risk Patient",
        "patient_data": {
            "name": "John Healthy",
            "age": "45",
            "gender": "Male",
            "blood_pressure": "120/80",
            "cholesterol": "180",
            "fasting_blood_sugar": "90",
            "resting_ecg": "Normal",
            "max_heart_rate": "170",
            "exercise_induced_angina": False,
            "st_depression": "0.2",
            "st_slope": "Upsloping",
            "num_major_vessels": "0",
            "thalassemia": "Normal"
        },
        "expected_risk": "Low"
    },
    {
        "name": "Medium Risk Patient",
        "patient_data": {
            "name": "Mary Moderate",
            "age": "58",
            "gender": "Female",
            "blood_pressure": "140/90",
            "cholesterol": "220",
            "fasting_blood_sugar": "110",
            "resting_ecg": "ST-T Wave Abnormality",
            "max_heart_rate": "150",
            "exercise_induced_angina": True,
            "st_depression": "1.0",
            "st_slope": "Flat",
            "num_major_vessels": "1",
            "thalassemia": "Fixed Defect",
            "prior_cardiac_event": {
                "type": "Angina",
                "time_since_event": "24",
                "severity": "Mild",
                "location": "Lateral"
            }
        },
        "expected_risk": "Medium"
    },
    {
        "name": "High Risk Patient",
        "patient_data": {
            "name": "Robert Critical",
            "age": "72",
            "gender": "Male",
            "blood_pressure": "160/95",
            "cholesterol": "260",
            "fasting_blood_sugar": "140",
            "resting_ecg": "Left Ventricular Hypertrophy",
            "max_heart_rate": "120",
            "exercise_induced_angina": True,
            "st_depression": "2.5",
            "st_slope": "Downsloping",
            "num_major_vessels": "3",
            "thalassemia": "Reversible Defect",
            "prior_cardiac_event": {
                "type": "Myocardial Infarction",
                "time_since_event": "6",
                "severity": "Severe",
                "location": "Anterior"
            }
        },
        "expected_risk": "High"
    },
    {
        "name": "Borderline Low-Medium Risk",
        "patient_data": {
            "name": "Sarah Borderline",
            "age": "52",
            "gender": "Female",
            "blood_pressure": "135/85",
            "cholesterol": "210",
            "fasting_blood_sugar": "105",
            "resting_ecg": "Normal",
            "max_heart_rate": "155",
            "exercise_induced_angina": False,
            "st_depression": "0.8",
            "st_slope": "Flat",
            "num_major_vessels": "1",
            "thalassemia": "Normal",
            "biomarkers": {
                "nt_probnp": "100",
                "troponin": "0.01",
                "crp": "2.5"
            }
        },
        "expected_risk": "Low-Medium Borderline"
    },
    {
        "name": "Borderline Medium-High Risk",
        "patient_data": {
            "name": "Tom Borderline",
            "age": "65",
            "gender": "Male",
            "blood_pressure": "150/92",
            "cholesterol": "240",
            "fasting_blood_sugar": "130",
            "resting_ecg": "ST-T Wave Abnormality",
            "max_heart_rate": "135",
            "exercise_induced_angina": True,
            "st_depression": "1.8",
            "st_slope": "Flat",
            "num_major_vessels": "2",
            "thalassemia": "Fixed Defect",
            "prior_cardiac_event": {
                "type": "Unstable Angina",
                "time_since_event": "12",
                "severity": "Moderate",
                "location": "Inferior"
            },
            "biomarkers": {
                "nt_probnp": "300",
                "troponin": "0.04",
                "crp": "5.0"
            }
        },
        "expected_risk": "Medium-High Borderline"
    }
]

def make_prediction(patient_data):
    """
    Make a prediction using the API
    """
    url = "http://localhost:8083/api/predict"
    headers = {"Content-Type": "application/json"}

    # Add a unique timestamp to each request to ensure different ECG seeds
    patient_data['timestamp'] = datetime.now().isoformat()

    # Enable debug mode
    patient_data['debug_mode'] = True

    data = {"patient_data": patient_data}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def main():
    """
    Main function to run tests
    """
    print("Testing Risk Assessment Accuracy")
    print("=" * 50)

    results = []

    for test_case in TEST_CASES:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 30)

        # Add a small delay between test cases to ensure different timestamps
        time.sleep(1)

        # Make prediction
        result = make_prediction(test_case['patient_data'])

        if result:
            # Extract prediction and risk category
            prediction = result.get('prediction', 0)
            risk_category = result.get('risk_category', 'Unknown')

            # Get risk thresholds
            risk_explanation = result.get('risk_explanation', {})
            thresholds = risk_explanation.get('thresholds', {}).get('final', {})
            low_medium = thresholds.get('low_medium', 0.15)
            medium_high = thresholds.get('medium_high', 0.35)

            # Determine if prediction matches expected risk
            expected_risk = test_case['expected_risk']
            matches_expected = False

            if expected_risk == "Low" and risk_category == "Low":
                matches_expected = True
            elif expected_risk == "Medium" and risk_category == "Medium":
                matches_expected = True
            elif expected_risk == "High" and risk_category == "High":
                matches_expected = True
            elif expected_risk == "Low-Medium Borderline":
                # For borderline cases, check if prediction is close to threshold
                if abs(prediction - low_medium) < 0.05:
                    matches_expected = True
            elif expected_risk == "Medium-High Borderline":
                # For borderline cases, check if prediction is close to threshold
                if abs(prediction - medium_high) < 0.05:
                    matches_expected = True

            # Print results
            print(f"Prediction: {prediction:.4f}")
            print(f"Risk Category: {risk_category}")
            print(f"Thresholds: Low-Medium = {low_medium:.4f}, Medium-High = {medium_high:.4f}")
            print(f"Expected Risk: {expected_risk}")
            print(f"Matches Expected: {'Yes' if matches_expected else 'No'}")

            # Store results
            results.append({
                "name": test_case['name'],
                "prediction": prediction,
                "risk_category": risk_category,
                "thresholds": {
                    "low_medium": low_medium,
                    "medium_high": medium_high
                },
                "expected_risk": expected_risk,
                "matches_expected": matches_expected
            })
        else:
            print("Failed to get prediction")

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
    results_file = f"risk_assessment_test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
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
