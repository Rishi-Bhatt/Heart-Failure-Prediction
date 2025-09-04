"""
Test ECG Abnormality Detection

This script tests the ECG abnormality detection functionality without modifying the existing code.
"""

import sys
import json
import numpy as np
from datetime import datetime

# Import ECG generator functions
from utils.ecg_generator import generate_ecg, analyze_ecg

def test_abnormality_detection(patient_data, expected_abnormalities=None):
    """
    Test abnormality detection for a given patient
    
    Parameters:
    -----------
    patient_data : dict
        Patient data dictionary
    expected_abnormalities : list
        List of expected abnormality types
    """
    print(f"\nTesting abnormality detection for: {patient_data.get('name', 'Unknown')}")
    print("-" * 50)
    
    # Generate ECG signal
    ecg_signal, ecg_time = generate_ecg(patient_data)
    print(f"Generated ECG signal with {len(ecg_signal)} data points")
    
    # Analyze ECG for abnormalities
    abnormalities = analyze_ecg(ecg_signal, ecg_time, patient_data)
    
    # Print detected abnormalities
    print("\nDetected abnormalities:")
    has_abnormalities = False
    for abnormality_type, instances in abnormalities.items():
        if instances:
            has_abnormalities = True
            print(f"  {abnormality_type}: {len(instances)} instances")
            for instance in instances:
                print(f"    - {instance.get('description', 'No description')}")
    
    if not has_abnormalities:
        print("  No abnormalities detected")
    
    # Check if expected abnormalities were detected
    if expected_abnormalities:
        print("\nExpected abnormalities check:")
        for abnormality_type in expected_abnormalities:
            if abnormalities.get(abnormality_type, []):
                print(f"  ✓ {abnormality_type}: DETECTED")
            else:
                print(f"  ✗ {abnormality_type}: NOT DETECTED")
    
    return abnormalities

def main():
    """
    Main function to run tests
    """
    print("Testing ECG Abnormality Detection")
    print("=" * 50)
    
    # Test cases with different risk profiles
    test_cases = [
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
            "expected_abnormalities": []
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
            "expected_abnormalities": ["PVCs", "QT_prolongation", "Atrial_Fibrillation"]
        },
        {
            "name": "Arrhythmia Patient",
            "patient_data": {
                "name": "Alice Arrhythmia",
                "age": 65,
                "gender": "Female",
                "blood_pressure": "145/85",
                "cholesterol": 230,
                "fasting_blood_sugar": 110,
                "resting_ecg": "ST-T Wave Abnormality",
                "max_heart_rate": 160,
                "exercise_induced_angina": True,
                "st_depression": 1.2,
                "st_slope": "Flat",
                "num_major_vessels": 1,
                "thalassemia": "Fixed Defect",
                "prior_cardiac_event": {
                    "type": "Arrhythmia",
                    "time_since_event": 12,
                    "severity": "Moderate",
                    "location": "N/A"
                },
                "biomarkers": {
                    "nt_probnp": 200,
                    "troponin": 0.03,
                    "crp": 4.0
                },
                "patient_id": "arrhythmia_test_patient"
            },
            "expected_abnormalities": ["PVCs", "Atrial_Fibrillation"]
        },
        {
            "name": "Tachycardia Patient",
            "patient_data": {
                "name": "Tom Tachy",
                "age": 35,
                "gender": "Male",
                "blood_pressure": "130/85",
                "cholesterol": 190,
                "fasting_blood_sugar": 95,
                "resting_ecg": "Normal",
                "max_heart_rate": 190,
                "exercise_induced_angina": False,
                "st_depression": 0.5,
                "st_slope": "Upsloping",
                "num_major_vessels": 0,
                "thalassemia": "Normal",
                "biomarkers": {
                    "nt_probnp": 70,
                    "troponin": 0.01,
                    "crp": 2.0
                },
                "patient_id": "tachycardia_test_patient"
            },
            "expected_abnormalities": ["Tachycardia"]
        },
        {
            "name": "Bradycardia Patient",
            "patient_data": {
                "name": "Betty Brady",
                "age": 75,
                "gender": "Female",
                "blood_pressure": "150/90",
                "cholesterol": 210,
                "fasting_blood_sugar": 100,
                "resting_ecg": "Normal",
                "max_heart_rate": 90,
                "exercise_induced_angina": False,
                "st_depression": 0.8,
                "st_slope": "Flat",
                "num_major_vessels": 1,
                "thalassemia": "Normal",
                "biomarkers": {
                    "nt_probnp": 150,
                    "troponin": 0.02,
                    "crp": 3.0
                },
                "patient_id": "bradycardia_test_patient"
            },
            "expected_abnormalities": ["Bradycardia"]
        }
    ]
    
    # Run tests
    results = []
    for test_case in test_cases:
        print(f"\n\nTesting {test_case['name']}")
        print("=" * 50)
        
        # Test abnormality detection
        abnormalities = test_abnormality_detection(
            test_case["patient_data"], 
            test_case.get("expected_abnormalities", [])
        )
        
        # Check if any abnormalities were detected
        detected_abnormalities = []
        for abnormality_type, instances in abnormalities.items():
            if instances:
                detected_abnormalities.append(abnormality_type)
        
        # Store results
        results.append({
            "name": test_case["name"],
            "detected_abnormalities": detected_abnormalities,
            "expected_abnormalities": test_case.get("expected_abnormalities", []),
            "success": any(detected_abnormalities) if test_case.get("expected_abnormalities", []) else True
        })
    
    # Print summary
    print("\n\nSummary")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"Total Tests: {total_count}")
    print(f"Successful Tests: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    
    # Print detailed results
    print("\nDetailed Results:")
    for result in results:
        print(f"  {result['name']}:")
        print(f"    Expected: {', '.join(result['expected_abnormalities']) if result['expected_abnormalities'] else 'None'}")
        print(f"    Detected: {', '.join(result['detected_abnormalities']) if result['detected_abnormalities'] else 'None'}")
        print(f"    Success: {'Yes' if result['success'] else 'No'}")
    
    # Save results to file
    results_file = f"ecg_abnormality_test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
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
