"""
Configuration settings for the Heart Failure Prediction System.

This module contains configuration settings for the application, including
file paths, model parameters, and other constants.
"""

import os

# Base directory for data storage
DATA_DIR = 'data'

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Data paths
DATA_PATHS = {
    'patient_data_directory': os.path.join(DATA_DIR, 'patients'),
    'model_directory': os.path.join(DATA_DIR, 'models'),
    'ecg_data_directory': os.path.join(DATA_DIR, 'ecg'),
    'longitudinal_data_directory': os.path.join(DATA_DIR, 'longitudinal'),
    'forecast_data_directory': os.path.join(DATA_DIR, 'forecasts'),
    'retraining_data_directory': os.path.join(DATA_DIR, 'retraining'),
}

# Ensure all directories exist
for path in DATA_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    }
}

# API configuration
API_CONFIG = {
    'default_port': 8083,
    'debug': True,
    'host': '0.0.0.0',
    'cors_origins': '*'
}

# Forecasting parameters
FORECAST_PARAMS = {
    'default_horizon': 6,  # Default forecast horizon in months
    'max_horizon': 24,     # Maximum forecast horizon in months
    'confidence_level': 0.95  # Confidence level for prediction intervals
}
