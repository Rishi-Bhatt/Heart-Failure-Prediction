"""
Test script for the Random Forest model extension.

This script tests the Random Forest model implementation by:
1. Training the model with sample data
2. Making predictions with the trained model
3. Comparing the Random Forest model with the existing logistic regression model
"""
import os
import json
import numpy as np
from datetime import datetime

# Import the Random Forest extension
import ml_model_extensions

# Import the existing ML model for comparison
import clinical_ml_model

def create_test_data():
    """Create sample patient data for testing"""
    # Create 10 sample patients
    patients = []
    
    for i in range(10):
        # Alternate between high and low risk patients
        high_risk = i % 2 == 0
        
        # Create patient data
        patient = {
            'patient_id': f'test_patient_{i}',
            'timestamp': datetime.now().isoformat(),
            'patient_data': {
                'name': f'Test Patient {i}',
                'age': 60 + (i * 2) if high_risk else 40 + (i * 2),
                'gender': 'Male' if i % 3 != 0 else 'Female',
                'blood_pressure': f'{140 + i * 5}/90' if high_risk else f'{120 + i}/80',
                'cholesterol': 240 + i * 5 if high_risk else 180 + i * 2,
                'fasting_blood_sugar': 130 + i * 2 if high_risk else 90 + i,
                'max_heart_rate': 100 - i if high_risk else 150 - i,
                'exercise_induced_angina': high_risk,
                'st_depression': 2.0 + i * 0.1 if high_risk else 0.5 + i * 0.05,
                'slope_of_st': 'Downsloping' if high_risk else 'Upsloping',
                'number_of_major_vessels': 2 if high_risk else 0,
                'thalassemia': 'Reversible Defect' if high_risk else 'Normal',
                'biomarkers': {
                    'nt_probnp': 1200 + i * 100 if high_risk else 100 + i * 10
                }
            },
            'prediction': 0.8 + i * 0.01 if high_risk else 0.2 + i * 0.01,
            'confidence': 0.9
        }
        
        patients.append(patient)
    
    return patients

def test_random_forest():
    """Test the Random Forest model implementation"""
    print("Testing Random Forest model implementation...")
    
    # Create test data
    print("Creating test data...")
    test_patients = create_test_data()
    
    # Train the model
    print("\nTraining Random Forest model...")
    training_result = ml_model_extensions.train_random_forest_model(test_patients)
    
    print(f"Training result: {training_result['message']}")
    if 'metrics' in training_result:
        print(f"Training metrics: {training_result['metrics']}")
    
    # Make a prediction with a sample patient
    print("\nMaking prediction with Random Forest model...")
    sample_patient = test_patients[0]['patient_data']
    
    rf_probability, rf_confidence, rf_explanation = ml_model_extensions.predict_heart_failure(sample_patient)
    print(f"Random Forest prediction: {rf_probability:.4f}, confidence: {rf_confidence:.4f}")
    
    # Get top factors
    if rf_explanation and 'top_factors' in rf_explanation:
        print("\nTop factors influencing Random Forest prediction:")
        for feature, data in rf_explanation['top_factors']:
            print(f"  {feature}: value={data['value']:.4f}, importance={data['importance']:.4f}, contribution={data['contribution']:.4f}")
    
    # Compare with logistic regression
    print("\nComparing with logistic regression model...")
    lr_probability, lr_confidence, lr_explanation = clinical_ml_model.predict_heart_failure(sample_patient)
    print(f"Logistic Regression prediction: {lr_probability:.4f}, confidence: {lr_confidence:.4f}")
    
    # Compare predictions
    print("\nModel comparison:")
    print(f"Random Forest:      {rf_probability:.4f} (confidence: {rf_confidence:.4f})")
    print(f"Logistic Regression: {lr_probability:.4f} (confidence: {lr_confidence:.4f})")
    print(f"Difference:          {abs(rf_probability - lr_probability):.4f}")
    
    # Use the comparison function
    print("\nUsing model comparison function...")
    comparison = ml_model_extensions.compare_models(sample_patient)
    
    print(f"Agreement between models: {comparison['agreement']:.4f}")
    print(f"Recommended model: {comparison['recommendation']}")
    
    print("\nRandom Forest model test completed successfully!")

if __name__ == "__main__":
    test_random_forest()
