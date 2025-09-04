import os
import sys
import numpy as np
import pandas as pd

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting test...")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir())

try:
    print("\nImporting HeartFailureModel...")
    from models.heart_failure_model import HeartFailureModel
    print("Import successful!")
    
    print("\nInitializing model...")
    model = HeartFailureModel()
    print("Model initialized!")
    
    print("\nCreating sample patient data...")
    # Create a sample patient data
    patient_data = {
        'name': 'Test Patient',
        'age': 65,
        'gender': 'Male',
        'blood_pressure': '140/90',
        'cholesterol': 220,
        'fasting_blood_sugar': 110,
        'chest_pain_type': 'Typical Angina',
        'ecg_result': 'Normal',
        'max_heart_rate': 140,
        'exercise_induced_angina': True,
        'st_depression': 1.5,
        'slope_of_st': 'Flat',
        'number_of_major_vessels': 2,
        'thalassemia': 'Normal',
        'prior_cardiac_event': {
            'type': 'Myocardial Infarction',
            'time_since_event': 6,
            'severity': 'Moderate'
        },
        'medications': [
            {
                'type': 'Beta-blockers',
                'time_of_administration': 2
            },
            {
                'type': 'ACE inhibitors',
                'time_of_administration': 4
            }
        ]
    }
    print("Sample patient data created!")
    
    print("\nCreating sample abnormalities...")
    # Create sample abnormalities
    abnormalities = {
        'PVCs': [{'time': 2.5, 'duration': 0.2}],
        'Flatlines': [],
        'Tachycardia': [{'time': 5.0, 'duration': 1.0, 'rate': 120}],
        'Bradycardia': [],
        'QT_prolongation': [{'time': 7.0, 'duration': 0.5, 'interval': 0.48}],
        'Atrial_Fibrillation': []
    }
    print("Sample abnormalities created!")
    
    print("\nPreprocessing data...")
    # Preprocess data
    features = model.preprocess_data(patient_data, abnormalities)
    print("Data preprocessed!")
    
    print("\nMaking prediction...")
    # Make prediction
    prediction, confidence, shap_values = model.predict(features)
    print("Prediction made!")
    
    print(f"\nPrediction: {prediction:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Top 5 features by importance:")
    
    # Get top 5 features by importance
    feature_importance = list(zip(model.feature_names, shap_values['values']))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, value in feature_importance[:5]:
        print(f"  {feature}: {value:.4f}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"\nError: {str(e)}")
    import traceback
    traceback.print_exc()
