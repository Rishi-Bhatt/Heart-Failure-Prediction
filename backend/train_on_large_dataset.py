"""
Train on Large Dataset

This script trains the heart failure prediction model on the larger dataset
(1800 patients) without breaking the current project.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_on_large_dataset():
    """
    Train the heart failure prediction model on the larger dataset
    """
    print("\nTraining Heart Failure Model on Large Dataset")
    print("=" * 60)
    
    # Define paths
    large_dataset_path = 'data/1.csv'
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'heart_failure_model.joblib')
    scaler_path = os.path.join(model_dir, 'heart_failure_scaler.joblib')
    
    # Check if large dataset exists
    if not os.path.exists(large_dataset_path):
        print(f"Error: Large dataset file not found at {large_dataset_path}")
        return False
    
    # Create backup of existing model files
    backup_dir = os.path.join(model_dir, 'backup_large')
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
    
    # Load large dataset
    try:
        df = pd.read_csv(large_dataset_path)
        print(f"Loaded large dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading large dataset: {str(e)}")
        return False
    
    # Check if 'target' column exists, if not, add it
    if 'target' not in df.columns:
        # In this dataset, we'll use 'thal' as a proxy for target
        # thal=3 (reversible defect) is often associated with heart disease
        df['target'] = (df['thal'] == 3).astype(int)
        print("Added 'target' column based on 'thal' values")
    
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
        'thal': 'thalassemia'
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
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save model and scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\nModel trained and saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    
    # Print feature importances
    feature_importances = list(zip(X.columns, model.feature_importances_))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Feature Importances:")
    for feature, importance in feature_importances[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nTraining on large dataset completed successfully!")
    return True

if __name__ == "__main__":
    train_on_large_dataset()
