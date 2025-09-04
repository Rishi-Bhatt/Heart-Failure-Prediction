"""
Test script for the enhanced hybrid model with Random Forest integration.

This script tests the enhanced hybrid model by:
1. Making predictions with the hybrid model
2. Comparing the contributions of different model components
3. Retraining the hybrid model with sample data
"""
import os
import json
import numpy as np
from datetime import datetime

# Import the hybrid model
import hybrid_model

# Import test data creation function from Random Forest test
from test_random_forest import create_test_data

def test_hybrid_model():
    """Test the enhanced hybrid model with Random Forest integration"""
    print("Testing enhanced hybrid model with Random Forest integration...")
    
    # Create test data
    print("Creating test data...")
    test_patients = create_test_data()
    
    # Make a prediction with a sample patient
    print("\nMaking prediction with hybrid model...")
    sample_patient = test_patients[0]['patient_data']
    
    prediction, confidence, explanations = hybrid_model.predict_heart_failure(sample_patient)
    print(f"Hybrid prediction: {prediction:.4f}, confidence: {confidence:.4f}")
    
    # Check which models were used
    print("\nModels used in prediction:")
    if 'ensemble' in explanations and 'models_used' in explanations['ensemble']:
        models_used = explanations['ensemble']['models_used']
        for model, used in models_used.items():
            print(f"  {model}: {'Used' if used else 'Not used'}")
    
    # Check ensemble weights
    print("\nEnsemble weights:")
    if 'ensemble' in explanations and 'weights' in explanations['ensemble']:
        weights = explanations['ensemble']['weights']
        for model, weight in weights.items():
            print(f"  {model}: {weight:.2f}")
    
    # Check model agreement
    print("\nModel agreement:")
    if 'ensemble' in explanations and 'agreement' in explanations['ensemble']:
        print(f"  Agreement: {explanations['ensemble']['agreement']:.2f}")
    
    # Retrain the hybrid model
    print("\nRetraining hybrid model...")
    retrain_result = hybrid_model.retrain_model(test_patients)
    
    print(f"Retraining result: {retrain_result['message']}")
    print("\nUpdated ensemble weights:")
    for model, weight in retrain_result['ensemble_weights'].items():
        print(f"  {model}: {weight:.2f}")
    
    # Make another prediction after retraining
    print("\nMaking prediction after retraining...")
    prediction, confidence, explanations = hybrid_model.predict_heart_failure(sample_patient)
    print(f"Hybrid prediction after retraining: {prediction:.4f}, confidence: {confidence:.4f}")
    
    print("\nEnhanced hybrid model test completed successfully!")

if __name__ == "__main__":
    test_hybrid_model()
