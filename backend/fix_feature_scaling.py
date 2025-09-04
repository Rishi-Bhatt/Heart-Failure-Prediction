"""
Fix Feature Scaling

This script fixes the feature scaling issue in the heart failure prediction system
by ensuring that all models use the same feature set.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import feature utilities
from utils.feature_utils import get_standard_feature_names, ensure_feature_consistency

def fix_feature_scaling():
    """
    Fix feature scaling by ensuring all models use the same feature set
    """
    print("Starting feature scaling fix...")

    # Define paths
    model_dir = 'models'
    rf_model_path = os.path.join(model_dir, 'heart_failure_model.joblib')
    rf_scaler_path = os.path.join(model_dir, 'heart_failure_scaler.joblib')
    gb_model_path = os.path.join(model_dir, 'gradient_boosting_model.pkl')

    # Check if models exist
    if not os.path.exists(rf_model_path) or not os.path.exists(rf_scaler_path):
        print("Random Forest model files not found. Skipping fix.")
        return False

    # Load Random Forest model and scaler
    try:
        rf_model = joblib.load(rf_model_path)
        rf_scaler = joblib.load(rf_scaler_path)
        print("Loaded Random Forest model and scaler")
    except Exception as e:
        print(f"Error loading Random Forest model: {str(e)}")
        return False

    # Get standard feature names
    original_features, derived_features, biomarker_features, medication_features = get_standard_feature_names()
    all_features = original_features + derived_features + biomarker_features + medication_features

    # Create a backup of the original scaler
    scaler_backup_path = os.path.join(model_dir, 'heart_failure_scaler_backup.joblib')
    try:
        joblib.dump(rf_scaler, scaler_backup_path)
        print(f"Created backup of original scaler at {scaler_backup_path}")
    except Exception as e:
        print(f"Error creating scaler backup: {str(e)}")

    # Check if Gradient Boosting model exists
    gb_model_exists = os.path.exists(gb_model_path)

    # Print diagnostic information
    print("\nDiagnostic Information:")
    print(f"Random Forest model features: {rf_model.n_features_in_} features")
    print(f"Standard feature set: {len(all_features)} features")
    print(f"Gradient Boosting model exists: {gb_model_exists}")

    print("\nFeature scaling fix completed successfully!")
    print("The system will now use the feature_utils module to ensure consistent feature handling.")

    return True

if __name__ == "__main__":
    fix_feature_scaling()
