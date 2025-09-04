"""
Test Feature Utilities

This script tests the feature utility functions to ensure they work correctly.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import feature utilities
from utils.feature_utils import get_standard_feature_names, calculate_derived_features, ensure_feature_consistency

# Create a test patient
test_patient = {
    "age": 65,
    "sex": 1,
    "chest_pain_type": 0,
    "resting_bp": 140,
    "cholesterol": 220,
    "fasting_blood_sugar": 0,
    "resting_ecg": 0,
    "max_heart_rate": 140,
    "exercise_induced_angina": 0,
    "st_depression": 0.5,
    "st_slope": 1,
    "num_major_vessels": 0,
    "thalassemia": 0,
    "prior_event_severity": 0,
    "time_since_event": 24,
    "pvc_count": 0,
    "qt_prolongation": 0,
    "af_detected": 0,
    "tachycardia_detected": 1,
    "bradycardia_detected": 0
}

# Test get_standard_feature_names
print("\nTesting get_standard_feature_names():")
original_features, derived_features, biomarker_features, medication_features = get_standard_feature_names()
print(f"Original features: {len(original_features)} features")
print(f"Derived features: {len(derived_features)} features")
print(f"Biomarker features: {len(biomarker_features)} features")
print(f"Medication features: {len(medication_features)} features")

# Test calculate_derived_features
print("\nTesting calculate_derived_features():")
derived_dict = calculate_derived_features(test_patient)
print(f"Derived features: {derived_dict}")

# Test ensure_feature_consistency
print("\nTesting ensure_feature_consistency():")
# Create a DataFrame with only original features
df = pd.DataFrame([test_patient])
print(f"Original DataFrame shape: {df.shape}")

# Ensure feature consistency
consistent_df = ensure_feature_consistency(df)
print(f"Consistent DataFrame shape: {consistent_df.shape}")
print(f"All features: {list(consistent_df.columns)}")

# Check if all required features are present
original_features, derived_features, biomarker_features, medication_features = get_standard_feature_names()
all_features = original_features + derived_features + biomarker_features + medication_features
missing_features = [f for f in all_features if f not in consistent_df.columns]
print(f"Missing features: {missing_features}")

print("\nTest completed successfully!")
