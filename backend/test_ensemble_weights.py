"""
Test Ensemble Weights

This script tests different ensemble weights to find the optimal combination
for accurate risk prediction across all risk levels.
"""

import os
import sys
import numpy as np
import pandas as pd
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import heart failure model
from models.heart_failure_model import HeartFailureModel
from utils.ecg_generator import generate_ecg, analyze_ecg

def test_ensemble_weights():
    """
    Test different ensemble weights
    """
    print("\nTesting Different Ensemble Weights")
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
    
    # Define different weight combinations to test
    weight_combinations = [
        {
            "name": "Original Weights",
            "weights": {
                "rule_based": 0.40,
                "logistic_regression": 0.30,
                "random_forest": 0.30
            }
        },
        {
            "name": "RF Focus",
            "weights": {
                "rule_based": 0.20,
                "logistic_regression": 0.10,
                "random_forest": 0.70
            }
        },
        {
            "name": "Rule-Based Focus",
            "weights": {
                "rule_based": 0.60,
                "logistic_regression": 0.20,
                "random_forest": 0.20
            }
        },
        {
            "name": "Balanced",
            "weights": {
                "rule_based": 0.33,
                "logistic_regression": 0.33,
                "random_forest": 0.34
            }
        }
    ]
    
    # Save weights to ensemble_optimizer.py
    weights_path = 'data/ensemble_weights.json'
    os.makedirs('data', exist_ok=True)
    
    # Test each weight combination
    results = []
    
    for weight_config in weight_combinations:
        print(f"\n\nTesting Weight Configuration: {weight_config['name']}")
        print("-" * 60)
        
        # Save weights to file
        with open(weights_path, 'w') as f:
            json.dump(weight_config['weights'], f)
        
        # Test each patient
        patient_results = []
        
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
                
                # Store results
                patient_results.append({
                    "patient_name": patient['name'],
                    "prediction": prediction,
                    "confidence": confidence,
                    "risk_category": risk_category
                })
                
            except Exception as e:
                print(f"Error during test: {str(e)}")
                patient_results.append({
                    "patient_name": patient['name'],
                    "error": str(e)
                })
        
        # Store results for this weight configuration
        results.append({
            "weight_config": weight_config,
            "patient_results": patient_results
        })
    
    # Print summary of results
    print("\n\nSummary of Results")
    print("=" * 60)
    
    for result in results:
        print(f"\nWeight Configuration: {result['weight_config']['name']}")
        print(f"Weights: {result['weight_config']['weights']}")
        print("-" * 40)
        
        for patient_result in result['patient_results']:
            if 'error' in patient_result:
                print(f"  {patient_result['patient_name']}: Error - {patient_result['error']}")
            else:
                print(f"  {patient_result['patient_name']}: {patient_result['prediction']:.4f} ({patient_result['risk_category']})")
    
    # Determine best weight configuration
    best_config = None
    best_score = -1
    
    for result in results:
        # Skip if any patient had an error
        if any('error' in patient_result for patient_result in result['patient_results']):
            continue
        
        # Calculate score based on correct risk categorization
        score = 0
        for patient_result in result['patient_results']:
            patient_name = patient_result['patient_name']
            risk_category = patient_result['risk_category']
            
            # Check if risk category matches expected category
            if patient_name == "Low Risk Patient" and risk_category == "Low":
                score += 1
            elif patient_name == "Medium Risk Patient" and risk_category == "Medium":
                score += 1
            elif patient_name == "High Risk Patient" and risk_category == "High":
                score += 1
        
        # Update best configuration if this one is better
        if score > best_score:
            best_score = score
            best_config = result['weight_config']
    
    # Print best weight configuration
    if best_config:
        print(f"\nBest Weight Configuration: {best_config['name']}")
        print(f"Weights: {best_config['weights']}")
        print(f"Score: {best_score}/3")
        
        # Save best weights to file
        with open(weights_path, 'w') as f:
            json.dump(best_config['weights'], f)
        print(f"Best weights saved to {weights_path}")
    else:
        print("\nNo valid weight configuration found")
    
    print("\nEnsemble weights test completed!")

if __name__ == "__main__":
    test_ensemble_weights()
