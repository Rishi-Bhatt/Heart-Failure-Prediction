"""
Feature Utilities

This module provides utility functions for feature handling and preprocessing
to ensure consistency across different models.
"""

import numpy as np
import pandas as pd

def get_standard_feature_names():
    """
    Get the standard feature names used across all models

    Returns:
    --------
    original_features : list
        List of original feature names
    derived_features : list
        List of derived feature names
    biomarker_features : list
        List of biomarker feature names
    medication_features : list
        List of medication feature names
    """
    # Original features
    original_features = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
        'exercise_induced_angina', 'st_depression', 'st_slope',
        'num_major_vessels', 'thalassemia', 'prior_event_severity',
        'time_since_event', 'pvc_count', 'qt_prolongation',
        'af_detected', 'tachycardia_detected', 'bradycardia_detected'
    ]

    # Derived features
    derived_features = [
        'age_squared', 'bmi', 'bp_age_ratio', 'cholesterol_hdl_ratio', 'heart_rate_recovery'
    ]

    # Biomarker features
    biomarker_features = [
        'nt_probnp', 'troponin', 'crp', 'bnp', 'creatinine'
    ]

    # Medication features
    medication_features = [
        'ace_inhibitor', 'arb', 'beta_blocker', 'statin',
        'antiplatelet', 'diuretic', 'calcium_channel_blocker'
    ]

    return original_features, derived_features, biomarker_features, medication_features

def get_model_feature_names():
    """
    Get the feature names used by the model trained on the large dataset

    Returns:
    --------
    feature_names : list
        List of feature names used by the model
    """
    # Get all feature categories
    original_features, derived_features, biomarker_features, medication_features = get_standard_feature_names()

    # Combine all features
    feature_names = original_features + derived_features + biomarker_features + medication_features

    return feature_names

def calculate_derived_features(features_dict):
    """
    Calculate derived features from original features

    Parameters:
    -----------
    features_dict : dict
        Dictionary containing original features

    Returns:
    --------
    derived_dict : dict
        Dictionary containing derived features
    """
    derived_dict = {}

    # Age squared
    try:
        age = float(features_dict.get('age', 60))
        derived_dict['age_squared'] = age ** 2
    except (ValueError, TypeError):
        derived_dict['age_squared'] = 3600  # Default for age 60

    # BP to age ratio
    try:
        resting_bp = float(features_dict.get('resting_bp', 120))
        age = float(features_dict.get('age', 60))
        derived_dict['bp_age_ratio'] = resting_bp / age if age > 0 else 2.0
    except (ValueError, TypeError):
        derived_dict['bp_age_ratio'] = 2.0

    # Heart rate recovery
    try:
        max_heart_rate = float(features_dict.get('max_heart_rate', 150))
        derived_dict['heart_rate_recovery'] = max(0, max_heart_rate - 120) / 10
    except (ValueError, TypeError):
        derived_dict['heart_rate_recovery'] = 3.0

    # Cholesterol HDL ratio
    try:
        cholesterol = float(features_dict.get('cholesterol', 200))
        hdl = float(features_dict.get('hdl', 50))
        derived_dict['cholesterol_hdl_ratio'] = cholesterol / hdl if hdl > 0 else 4.0
    except (ValueError, TypeError):
        derived_dict['cholesterol_hdl_ratio'] = 4.0

    # BMI
    try:
        weight = float(features_dict.get('weight', 75))
        height = float(features_dict.get('height', 1.75))
        derived_dict['bmi'] = weight / (height ** 2) if height > 0 else 25.0
    except (ValueError, TypeError):
        derived_dict['bmi'] = 25.0

    return derived_dict

def calculate_biomarker_risk(biomarkers):
    """
    Calculate risk score based on biomarker values

    Parameters:
    -----------
    biomarkers : dict
        Dictionary containing biomarker values

    Returns:
    --------
    risk_score : float
        Risk score based on biomarkers (0-1)
    """
    risk_score = 0.0

    # NT-proBNP (pg/mL)
    # Normal: <125 for patients <75 years, <450 for patients ≥75 years
    # Elevated: >450 for patients <50 years, >900 for patients 50-75 years, >1800 for patients >75 years
    try:
        nt_probnp = float(biomarkers.get('nt_probnp', 0))
        age = float(biomarkers.get('age', 60))

        if age < 50 and nt_probnp > 450:
            risk_score += 0.30
        elif 50 <= age < 75 and nt_probnp > 900:
            risk_score += 0.30
        elif age >= 75 and nt_probnp > 1800:
            risk_score += 0.30
        elif nt_probnp > 300:  # Moderately elevated
            risk_score += 0.15
    except (ValueError, TypeError):
        pass

    # Troponin (ng/mL)
    # Normal: <0.04
    # Elevated: >0.04
    try:
        troponin = float(biomarkers.get('troponin', 0))
        if troponin > 0.1:  # Significantly elevated
            risk_score += 0.35
        elif troponin > 0.04:  # Moderately elevated
            risk_score += 0.20
    except (ValueError, TypeError):
        pass

    # CRP (mg/L)
    # Normal: <3.0
    # Elevated: >3.0
    try:
        crp = float(biomarkers.get('crp', 0))
        if crp > 10.0:  # Significantly elevated
            risk_score += 0.20
        elif crp > 3.0:  # Moderately elevated
            risk_score += 0.10
    except (ValueError, TypeError):
        pass

    # BNP (pg/mL)
    # Normal: <100
    # Elevated: >100
    try:
        bnp = float(biomarkers.get('bnp', 0))
        if bnp > 400:  # Significantly elevated
            risk_score += 0.25
        elif bnp > 100:  # Moderately elevated
            risk_score += 0.15
    except (ValueError, TypeError):
        pass

    # Creatinine (mg/dL)
    # Normal: 0.7-1.3 for men, 0.6-1.1 for women
    # Elevated: >1.3 for men, >1.1 for women
    try:
        creatinine = float(biomarkers.get('creatinine', 0))
        sex = int(biomarkers.get('sex', 1))  # 1 for male, 0 for female

        if sex == 1 and creatinine > 1.5:  # Male, significantly elevated
            risk_score += 0.20
        elif sex == 1 and creatinine > 1.3:  # Male, moderately elevated
            risk_score += 0.10
        elif sex == 0 and creatinine > 1.3:  # Female, significantly elevated
            risk_score += 0.20
        elif sex == 0 and creatinine > 1.1:  # Female, moderately elevated
            risk_score += 0.10
    except (ValueError, TypeError):
        pass

    # Normalize risk score to 0-1 range
    # Maximum possible score from all biomarkers is approximately 1.3
    normalized_risk = min(1.0, risk_score / 1.3)

    return normalized_risk

def calculate_medication_effect(medications, base_risk):
    """
    Calculate the effect of medications on the risk score

    Parameters:
    -----------
    medications : dict
        Dictionary containing medication information (1 if taking, 0 if not)
    base_risk : float
        Base risk score before medication adjustment

    Returns:
    --------
    adjusted_risk : float
        Risk score adjusted for medication effects (0-1)
    """
    # Start with the base risk
    adjusted_risk = base_risk
    risk_reduction = 0.0

    # ACE inhibitors - reduce risk by 10-15%
    if medications.get('ace_inhibitor', 0) == 1:
        risk_reduction += 0.12

    # ARBs - reduce risk by 10-15%
    if medications.get('arb', 0) == 1:
        risk_reduction += 0.12

    # Beta blockers - reduce risk by 15-20%
    if medications.get('beta_blocker', 0) == 1:
        risk_reduction += 0.18

    # Statins - reduce risk by 15-25%
    if medications.get('statin', 0) == 1:
        risk_reduction += 0.20

    # Antiplatelet drugs - reduce risk by 10-15%
    if medications.get('antiplatelet', 0) == 1:
        risk_reduction += 0.12

    # Diuretics - reduce risk by 5-10%
    if medications.get('diuretic', 0) == 1:
        risk_reduction += 0.08

    # Calcium channel blockers - reduce risk by 5-10%
    if medications.get('calcium_channel_blocker', 0) == 1:
        risk_reduction += 0.08

    # Cap the maximum risk reduction at 60%
    risk_reduction = min(0.6, risk_reduction)

    # Apply the risk reduction
    adjusted_risk = base_risk * (1 - risk_reduction)

    return adjusted_risk

def ensure_feature_consistency(features_df):
    """
    Ensure that all required features are present in the DataFrame

    Parameters:
    -----------
    features_df : pandas.DataFrame
        DataFrame containing features

    Returns:
    --------
    consistent_df : pandas.DataFrame
        DataFrame with all required features
    """
    original_features, derived_features, biomarker_features, medication_features = get_standard_feature_names()
    all_features = original_features + derived_features + biomarker_features + medication_features

    # Create a copy to avoid modifying the original
    consistent_df = features_df.copy()

    # Ensure all original features are present
    for feature in original_features:
        if feature not in consistent_df.columns:
            consistent_df[feature] = 0

    # Calculate derived features if missing
    if not all(feature in consistent_df.columns for feature in derived_features):
        # Extract first row as a dictionary
        first_row = consistent_df.iloc[0].to_dict()

        # Calculate derived features
        derived_dict = calculate_derived_features(first_row)

        # Add derived features to DataFrame
        for feature, value in derived_dict.items():
            if feature not in consistent_df.columns:
                consistent_df[feature] = value

    # Add default values for biomarker features if missing
    for feature in biomarker_features:
        if feature not in consistent_df.columns:
            consistent_df[feature] = 0

    # Add default values for medication features if missing
    for feature in medication_features:
        if feature not in consistent_df.columns:
            consistent_df[feature] = 0

    # Ensure correct order
    return consistent_df[all_features]
