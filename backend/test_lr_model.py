"""
Simple test script for the logistic regression model.
"""
import os
import sys
import traceback
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting logistic regression model test...")

try:
    print("Importing ClinicallyInformedLogisticRegression...")
    from clinical_ml_model import ClinicallyInformedLogisticRegression, engineer_clinical_features
    
    print("Creating test patient data...")
    test_patient = {
        'age': 60,
        'gender': 'Male',
        'blood_pressure': '140/90',
        'cholesterol': 240,
        'fasting_blood_sugar': 130,
        'max_heart_rate': 100,
        'exercise_induced_angina': True,
        'st_depression': 2.0,
        'slope_of_st': 'Downsloping',
        'number_of_major_vessels': 2,
        'thalassemia': 'Reversible Defect',
        'biomarkers': {
            'nt_probnp': 1200
        }
    }
    
    print("Engineering features...")
    features = engineer_clinical_features(test_patient)
    feature_vector = np.array([[v for v in features.values()]])
    
    print("Creating model...")
    model = ClinicallyInformedLogisticRegression()
    
    print("Loading model...")
    if model.load_model():
        print("Model loaded successfully!")
        
        print("Making prediction...")
        try:
            probability = model.predict_proba(feature_vector)[0, 1]
            print(f"Prediction probability: {probability:.4f}")
            
            print("Getting explanation...")
            explanation = model.explain_prediction(features)
            print(f"Explanation generated with {len(explanation.get('top_factors', []))} top factors")
            
            print("Test completed successfully!")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            traceback.print_exc()
    else:
        print("Failed to load model.")

except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()

print("Test completed.")
