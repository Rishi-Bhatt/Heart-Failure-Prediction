"""
Retrain Model

This script retrains the heart failure prediction model with the optimized parameters.
"""

import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import heart failure model
from models.heart_failure_model import HeartFailureModel

def retrain_model():
    """
    Retrain the heart failure prediction model
    """
    print("\nRetraining Heart Failure Prediction Model")
    print("=" * 60)
    
    # Create a heart failure model instance
    model = HeartFailureModel()
    
    # Force retraining by deleting existing model files
    if os.path.exists(model.model_path):
        os.remove(model.model_path)
        print(f"Removed existing model: {model.model_path}")
    
    if os.path.exists(model.scaler_path):
        os.remove(model.scaler_path)
        print(f"Removed existing scaler: {model.scaler_path}")
    
    # Retrain the model
    print("\nTraining new model with optimized parameters...")
    model.download_dataset()
    model.train_model()
    
    print("\nModel retraining completed!")

if __name__ == "__main__":
    retrain_model()
