"""
Test Gradient Boosting Integration

This script tests if gradient boosting is being used in the risk assessment
without modifying the existing code.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
from models.heart_failure_model import HeartFailureModel
from utils.ecg_generator import generate_ecg, analyze_ecg

# Try to import gradient boosting model
try:
    from models.gradient_boosting_model import gb_model, train_gradient_boosting_model
    GRADIENT_BOOSTING_AVAILABLE = True
    print("Gradient Boosting model imported successfully")
except ImportError:
    print("Warning: Gradient Boosting model not found. Will not use it in ensemble.")
    GRADIENT_BOOSTING_AVAILABLE = False

# Import ensemble optimizer
try:
    from ensemble_optimizer import get_current_weights, DEFAULT_WEIGHTS, DEFAULT_WEIGHTS_WITH_GB
    ENSEMBLE_OPTIMIZER_AVAILABLE = True
    print("Ensemble optimizer imported successfully")
except ImportError:
    print("Warning: ensemble_optimizer module not found. Using default weights.")
    ENSEMBLE_OPTIMIZER_AVAILABLE = False
    DEFAULT_WEIGHTS = {'rule_based': 0.40, 'logistic_regression': 0.30, 'random_forest': 0.30}
    DEFAULT_WEIGHTS_WITH_GB = {
        'rule_based': 0.30,
        'logistic_regression': 0.20,
        'random_forest': 0.25,
        'gradient_boosting': 0.25
    }

def check_gradient_boosting_integration():
    """
    Check if gradient boosting is being used in the risk assessment
    """
    print("\nChecking Gradient Boosting Integration")
    print("=" * 50)
    
    # Check if gradient boosting model is available
    print(f"Gradient Boosting Available: {GRADIENT_BOOSTING_AVAILABLE}")
    
    if GRADIENT_BOOSTING_AVAILABLE:
        # Check if gradient boosting model is trained
        print(f"Gradient Boosting Trained: {gb_model.is_trained}")
        
        # Check if gradient boosting is included in ensemble weights
        current_weights = get_current_weights() if ENSEMBLE_OPTIMIZER_AVAILABLE else DEFAULT_WEIGHTS
        print(f"Current Ensemble Weights: {current_weights}")
        
        # Check if gradient boosting is in the weights
        gb_in_weights = 'gradient_boosting' in current_weights
        print(f"Gradient Boosting in Weights: {gb_in_weights}")
        
        # If gradient boosting is not in weights, check if it can be added
        if not gb_in_weights:
            print("\nGradient Boosting is not currently in the ensemble weights.")
            print("This could be because:")
            print("1. The model has not been trained yet")
            print("2. The weights have not been updated to include gradient boosting")
            print("3. Gradient boosting is intentionally excluded from the ensemble")
            
            # Check if the model is trained
            if not gb_model.is_trained:
                print("\nThe gradient boosting model is not trained.")
                print("It needs to be trained before it can be used in the ensemble.")
            else:
                print("\nThe gradient boosting model is trained but not included in the weights.")
                print("The weights would need to be updated to include gradient boosting.")
                
            # Show what the weights would look like with gradient boosting
            print(f"\nDefault Weights with Gradient Boosting: {DEFAULT_WEIGHTS_WITH_GB}")
    else:
        print("Gradient Boosting is not available in this installation.")
        print("The system will use the default ensemble without gradient boosting.")
    
    # Check if gradient boosting is being used in the heart failure model
    print("\nChecking Heart Failure Model Code")
    print("=" * 50)
    
    # Create a heart failure model instance
    model = HeartFailureModel()
    
    # Create a test patient
    test_patient = {
        "name": "Test Patient",
        "age": 65,
        "gender": "Male",
        "blood_pressure": "140/90",
        "cholesterol": 220,
        "fasting_blood_sugar": 110,
        "resting_ecg": "Normal",
        "max_heart_rate": 140,
        "exercise_induced_angina": False,
        "st_depression": 0.5,
        "st_slope": "Flat",
        "num_major_vessels": 1,
        "thalassemia": "Normal"
    }
    
    # Generate ECG and analyze for abnormalities
    ecg_signal, ecg_time = generate_ecg(test_patient)
    abnormalities = analyze_ecg(ecg_signal, ecg_time, test_patient)
    
    # Preprocess data
    features = model.preprocess_data(test_patient, abnormalities)
    
    # Make prediction with debug mode
    prediction, confidence, shap_values = model.predict(features, debug=True)
    
    print(f"\nPrediction: {prediction:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Risk Category: {'High' if prediction > 0.35 else 'Medium' if prediction > 0.15 else 'Low'}")
    
    # Check if gradient boosting is mentioned in the model's predict method
    import inspect
    predict_source = inspect.getsource(model.predict)
    gb_in_predict = 'gradient_boosting' in predict_source
    print(f"\nGradient Boosting mentioned in predict method: {gb_in_predict}")
    
    # Count occurrences of gradient_boosting in the predict method
    gb_count = predict_source.count('gradient_boosting')
    print(f"Number of 'gradient_boosting' occurrences in predict method: {gb_count}")
    
    return {
        'gradient_boosting_available': GRADIENT_BOOSTING_AVAILABLE,
        'gradient_boosting_trained': gb_model.is_trained if GRADIENT_BOOSTING_AVAILABLE else False,
        'gradient_boosting_in_weights': gb_in_weights if GRADIENT_BOOSTING_AVAILABLE else False,
        'gradient_boosting_in_predict': gb_in_predict,
        'gradient_boosting_count': gb_count,
        'prediction': float(prediction),
        'confidence': float(confidence)
    }

def main():
    """
    Main function
    """
    print("Testing Gradient Boosting Integration")
    print("=" * 50)
    
    # Check gradient boosting integration
    results = check_gradient_boosting_integration()
    
    # Save results to file
    results_file = f"gradient_boosting_test_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\nSummary")
    print("=" * 50)
    print(f"Gradient Boosting Available: {results['gradient_boosting_available']}")
    print(f"Gradient Boosting Trained: {results['gradient_boosting_trained']}")
    print(f"Gradient Boosting in Weights: {results['gradient_boosting_in_weights']}")
    print(f"Gradient Boosting in Predict Method: {results['gradient_boosting_in_predict']}")
    print(f"Gradient Boosting Count in Predict: {results['gradient_boosting_count']}")
    print(f"Prediction: {results['prediction']:.4f}")
    print(f"Confidence: {results['confidence']:.4f}")
    
    # Conclusion
    print("\nConclusion")
    print("=" * 50)
    
    if results['gradient_boosting_available'] and results['gradient_boosting_in_predict']:
        if results['gradient_boosting_trained'] and results['gradient_boosting_in_weights']:
            print("Gradient Boosting is fully integrated and being used in the risk assessment.")
        elif results['gradient_boosting_trained']:
            print("Gradient Boosting is trained and the code supports it, but it's not included in the weights.")
            print("To use it, the weights need to be updated to include gradient_boosting.")
        else:
            print("Gradient Boosting is available and the code supports it, but the model is not trained.")
            print("To use it, the model needs to be trained first.")
    elif results['gradient_boosting_available']:
        print("Gradient Boosting is available but not being used in the risk assessment.")
        print("The predict method doesn't include gradient_boosting in the ensemble.")
    else:
        print("Gradient Boosting is not available in this installation.")
        print("The system is using the default ensemble without gradient boosting.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
