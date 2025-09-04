"""
Train Gradient Boosting Model

This script trains the gradient boosting model and updates the ensemble weights
to include it without modifying the existing code.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import gradient boosting model
try:
    from models.gradient_boosting_model import train_gradient_boosting_model, gb_model
    GRADIENT_BOOSTING_AVAILABLE = True
    print("Gradient Boosting model imported successfully")
except ImportError:
    print("Error: Gradient Boosting model not found. Cannot proceed.")
    sys.exit(1)

# Import ensemble optimizer
try:
    from ensemble_optimizer import get_current_weights, save_weights, DEFAULT_WEIGHTS_WITH_GB
    ENSEMBLE_OPTIMIZER_AVAILABLE = True
    print("Ensemble optimizer imported successfully")
except ImportError:
    print("Error: ensemble_optimizer module not found. Cannot proceed.")
    sys.exit(1)

def train_gradient_boosting():
    """
    Train the gradient boosting model and update ensemble weights
    """
    print("\nTraining Gradient Boosting Model")
    print("=" * 50)
    
    # Load patient data
    patient_files = [f for f in os.listdir('data/patients') if f.endswith('.json')]
    
    if len(patient_files) < 10:
        print(f"Insufficient data for training Gradient Boosting model. Need at least 10 records, but only found {len(patient_files)}.")
        return False
    
    # Load patient data
    patient_data_list = []
    for filename in patient_files:
        try:
            with open(f'data/patients/{filename}', 'r') as f:
                patient_data = json.load(f)
                patient_data_list.append(patient_data)
        except Exception as e:
            print(f"Error loading patient data from {filename}: {str(e)}")
            continue
    
    print(f"Loaded {len(patient_data_list)} patient records for Gradient Boosting model training")
    
    # Train the model
    training_result = train_gradient_boosting_model(patient_data_list)
    
    # Check if training was successful
    if not training_result.get('success', False):
        print(f"Error training Gradient Boosting model: {training_result.get('message', 'Unknown error')}")
        return False
    
    print(f"Gradient Boosting model trained successfully: {training_result.get('message', 'Success')}")
    
    # Update ensemble weights to include gradient boosting
    current_weights = get_current_weights()
    print(f"Current ensemble weights: {current_weights}")
    
    # Check if gradient boosting is already in weights
    if 'gradient_boosting' in current_weights:
        print("Gradient Boosting is already included in ensemble weights")
        return True
    
    # Add gradient boosting to weights
    new_weights = DEFAULT_WEIGHTS_WITH_GB.copy()
    print(f"New ensemble weights with Gradient Boosting: {new_weights}")
    
    # Save new weights
    save_weights(new_weights)
    print("Ensemble weights updated to include Gradient Boosting")
    
    return True

def main():
    """
    Main function
    """
    print("Training Gradient Boosting Model and Updating Ensemble Weights")
    print("=" * 50)
    
    # Train gradient boosting model and update weights
    success = train_gradient_boosting()
    
    if success:
        print("\nGradient Boosting model trained and ensemble weights updated successfully")
        print("The model will now use Gradient Boosting in the risk assessment")
    else:
        print("\nFailed to train Gradient Boosting model or update ensemble weights")
        print("The model will continue to use the existing ensemble without Gradient Boosting")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
