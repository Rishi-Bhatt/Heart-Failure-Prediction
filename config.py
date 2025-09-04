"""
Configuration settings for the Heart Failure Prediction System
"""

# Model parameters
MODEL_CONFIG = {
    # Rule-based model parameters
    'rule_based': {
        'age_threshold': 65,
        'bp_systolic_threshold': 140,
        'bp_diastolic_threshold': 90,
        'cholesterol_threshold': 200,
        'fasting_glucose_threshold': 100,
        'max_heart_rate_threshold': 120,
        'st_depression_threshold': 1.0,
        'num_vessels_threshold': 1,
    },
    
    # Logistic regression parameters
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 5000,
        'class_weight': 'balanced',
        'tol': 1e-5,
    },
    
    # Random forest parameters
    'random_forest': {
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'bootstrap': True,
        'oob_score': True,
    },
    
    # Hybrid model parameters
    'hybrid': {
        'base_weights': {
            'rule_based': 0.3,
            'logistic_regression': 0.3,
            'random_forest': 0.4,
        },
        'confidence_scaling': True,
    },
}

# NT-proBNP thresholds by age
NT_PROBNP_THRESHOLDS = {
    'age_lt_50': 450,
    'age_50_to_75': 900,
    'age_gt_75': 1800,
}

# Feature importance direction factors
FEATURE_DIRECTION_FACTORS = {
    'age': 1.0,
    'gender_male': 1.0,
    'systolic_bp': 1.0,
    'diastolic_bp': 1.0,
    'cholesterol': 1.0,
    'fasting_blood_sugar': 1.0,
    'max_heart_rate': -1.0,  # Lower is worse
    'exercise_angina': 1.0,
    'st_depression': 1.0,
    'st_slope_flat': 1.0,
    'st_slope_downsloping': 1.0,
    'num_vessels': 1.0,
    'thalassemia_fixed': 1.0,
    'thalassemia_reversible': 1.0,
    'nt_probnp': 1.0,
    'nt_probnp_normalized': 1.0,
    'prior_cardiac_event': 1.5,  # Higher weight for prior events
    'ecg_af_detected': 1.2,
    'ecg_pvc_count': 1.0,
    'ecg_qt_prolongation': 1.0,
}

# Longitudinal tracking parameters
LONGITUDINAL_CONFIG = {
    'min_data_points': 2,  # Minimum number of data points needed for trend analysis
    'max_time_gap': 365,   # Maximum gap in days between measurements to consider them related
    'decay_parameter': 0.1,  # Controls how quickly current prediction influence diminishes
    'trend_window': 3,     # Number of previous measurements to use for trend calculation
}

# Risk forecasting parameters
FORECASTING_CONFIG = {
    'time_horizons': [180, 365, 730],  # Forecast horizons in days (6 months, 1 year, 2 years)
    'confidence_level': 0.95,  # 95% confidence interval
    'intervention_effect_rates': {
        'medication': 0.01,  # Rate parameter for medication effect (per day)
        'lifestyle': 0.005,  # Rate parameter for lifestyle changes (per day)
        'combined': 0.015,   # Rate parameter for combined interventions (per day)
    },
    'intervention_max_impacts': {
        'medication': 0.15,  # Maximum risk reduction from medication
        'lifestyle': 0.10,   # Maximum risk reduction from lifestyle changes
        'combined': 0.25,    # Maximum risk reduction from combined interventions
    },
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8080,
    'debug': False,
    'threaded': True,
}

# Visualization configuration
VIZ_CONFIG = {
    'risk_colors': {
        'low': '#1a9850',      # Green
        'moderate': '#ffffbf',  # Yellow
        'high': '#d73027',     # Red
    },
    'risk_thresholds': {
        'low_moderate': 0.3,
        'moderate_high': 0.7,
    },
    'ecg_abnormality_colors': {
        'af': '#e41a1c',       # Red
        'pvc': '#377eb8',      # Blue
        'qt_prolongation': '#4daf4a',  # Green
    },
}

# Data paths
DATA_PATHS = {
    'heart_disease_uci': 'data/heart_disease_uci.csv',
    'synthetic_training': 'data/synthetic_training.csv',
    'nt_probnp_reference': 'data/nt_probnp_reference.csv',
    'ecg_samples': 'data/ecg_samples.csv',
    'validation_dataset': 'data/validation_dataset.csv',
    'model_directory': 'models/',
    'patient_data_directory': 'data/patients/',
}
