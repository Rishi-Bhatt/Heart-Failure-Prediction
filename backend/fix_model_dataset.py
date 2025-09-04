"""
Fix Model Dataset

This script fixes the model dataset issue by ensuring the model is trained
on the same features that are used for prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_model_dataset():
    """
    Fix the model dataset issue by training the model on the correct features
    """
    print("\nFixing Model Dataset Issue")
    print("=" * 60)
    
    # Define paths
    raw_data_path = 'data/raw_data.csv'
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'heart_failure_model.joblib')
    scaler_path = os.path.join(model_dir, 'heart_failure_scaler.joblib')
    
    # Check if raw data exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        return False
    
    # Create backup of existing model files
    backup_dir = os.path.join(model_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    
    if os.path.exists(model_path):
        backup_model_path = os.path.join(backup_dir, 'heart_failure_model.joblib')
        try:
            joblib.dump(joblib.load(model_path), backup_model_path)
            print(f"Created backup of model at {backup_model_path}")
        except Exception as e:
            print(f"Error creating model backup: {str(e)}")
    
    if os.path.exists(scaler_path):
        backup_scaler_path = os.path.join(backup_dir, 'heart_failure_scaler.joblib')
        try:
            joblib.dump(joblib.load(scaler_path), backup_scaler_path)
            print(f"Created backup of scaler at {backup_scaler_path}")
        except Exception as e:
            print(f"Error creating scaler backup: {str(e)}")
    
    # Load raw data
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Loaded raw data with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading raw data: {str(e)}")
        return False
    
    # Rename columns to match the feature names used in prediction
    column_mapping = {
        'age': 'age',
        'sex': 'sex',
        'cp': 'chest_pain_type',
        'trestbps': 'resting_bp',
        'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'resting_ecg',
        'thalach': 'max_heart_rate',
        'exang': 'exercise_induced_angina',
        'oldpeak': 'st_depression',
        'slope': 'st_slope',
        'ca': 'num_major_vessels',
        'thal': 'thalassemia',
        'target': 'target'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Add missing columns with default values
    df['prior_event_severity'] = 0
    df['time_since_event'] = 0
    df['pvc_count'] = 0
    df['qt_prolongation'] = 0
    df['af_detected'] = 0
    df['tachycardia_detected'] = 0
    df['bradycardia_detected'] = 0
    
    # Calculate derived features
    df['age_squared'] = df['age'] ** 2
    df['bmi'] = 25.0  # Default value
    df['bp_age_ratio'] = df['resting_bp'] / df['age']
    df['cholesterol_hdl_ratio'] = 4.0  # Default value
    df['heart_rate_recovery'] = (df['max_heart_rate'] - 120) / 10
    df['heart_rate_recovery'] = df['heart_rate_recovery'].clip(lower=0)
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train optimized Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel trained and saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Print feature names for reference
    print("\nFeature names used for training:")
    for i, feature in enumerate(X.columns):
        print(f"  {i+1}. {feature}")
    
    print("\nModel dataset fix completed successfully!")
    return True

if __name__ == "__main__":
    fix_model_dataset()
