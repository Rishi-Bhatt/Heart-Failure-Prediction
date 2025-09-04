"""
Risk Calibration Module for Heart Failure Prediction System

This module provides functions for calibrating risk thresholds and improving
differentiation between low, medium, and high risk levels.
"""

import numpy as np
import json
import os
from datetime import datetime

# Define default risk thresholds
DEFAULT_RISK_THRESHOLDS = {
    'low_medium': 0.15,  # Below this is low risk
    'medium_high': 0.35  # Above this is high risk
}

# Path to save calibration data
CALIBRATION_FILE = 'data/risk_calibration.json'

# Ensure these values are used if no calibration file exists
if not os.path.exists(CALIBRATION_FILE):
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(DEFAULT_RISK_THRESHOLDS, f, indent=2)

def get_risk_thresholds():
    """
    Get the current risk thresholds

    Returns:
    --------
    dict
        Dictionary containing risk thresholds
    """
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_RISK_THRESHOLDS
    else:
        return DEFAULT_RISK_THRESHOLDS

def save_risk_thresholds(thresholds):
    """
    Save risk thresholds to file

    Parameters:
    -----------
    thresholds : dict
        Dictionary containing risk thresholds
    """
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(thresholds, f, indent=2)

def get_age_adjusted_thresholds(age, gender='Unknown'):
    """
    Get age and gender adjusted risk thresholds

    Parameters:
    -----------
    age : int or float
        Patient age
    gender : str
        Patient gender ('Male', 'Female', or 'Unknown')

    Returns:
    --------
    dict
        Dictionary containing adjusted risk thresholds
    """
    # Get base thresholds
    thresholds = get_risk_thresholds()

    # Age adjustment factor
    # Younger patients need higher thresholds to be considered at risk
    # Older patients need lower thresholds to be considered at risk
    if age < 40:
        age_factor = 1.2  # Increase threshold for young patients
    elif age < 55:
        age_factor = 1.1  # Slight increase for middle-aged patients
    elif age < 70:
        age_factor = 0.9  # Slight decrease for older patients
    else:
        age_factor = 0.8  # Decrease threshold for elderly patients

    # Ensure the adjustment doesn't make the threshold too high
    if thresholds['low_medium'] * age_factor > 0.25:
        age_factor = 0.25 / thresholds['low_medium']
    if thresholds['medium_high'] * age_factor > 0.5:
        age_factor = 0.5 / thresholds['medium_high']

    # Gender adjustment factor
    # Women typically have lower risk at the same age compared to men
    if gender == 'Female':
        gender_factor = 1.1  # Increase threshold for women
    elif gender == 'Male':
        gender_factor = 0.9  # Decrease threshold for men
    else:
        gender_factor = 1.0  # No adjustment for unknown gender

    # Calculate adjusted thresholds
    adjusted_thresholds = {
        'low_medium': thresholds['low_medium'] * age_factor * gender_factor,
        'medium_high': thresholds['medium_high'] * age_factor * gender_factor
    }

    return adjusted_thresholds

def get_biomarker_adjusted_thresholds(thresholds, biomarkers):
    """
    Adjust risk thresholds based on biomarker values

    Parameters:
    -----------
    thresholds : dict
        Base risk thresholds
    biomarkers : dict
        Dictionary containing biomarker values

    Returns:
    --------
    dict
        Dictionary containing adjusted risk thresholds
    """
    if not biomarkers:
        return thresholds

    # Copy thresholds to avoid modifying the original
    adjusted_thresholds = thresholds.copy()

    # Track the total adjustment factor
    total_adjustment_factor = 1.0

    # NT-proBNP adjustment
    try:
        nt_probnp_value = biomarkers.get('nt_probnp', '')
        # Handle both string and numeric values
        if isinstance(nt_probnp_value, str) and nt_probnp_value.strip():
            nt_probnp = float(nt_probnp_value)
        elif isinstance(nt_probnp_value, (int, float)):
            nt_probnp = float(nt_probnp_value)
        else:
            nt_probnp = 0

        if nt_probnp > 450:  # High NT-proBNP
            # Lower thresholds to increase risk
            total_adjustment_factor *= 0.85
        elif nt_probnp > 125:  # Elevated NT-proBNP
            # Slightly lower thresholds
            total_adjustment_factor *= 0.95
    except (ValueError, TypeError):
        # If NT-proBNP value is invalid, skip adjustment
        pass

    # Troponin adjustment
    try:
        troponin_value = biomarkers.get('troponin', '')
        # Handle both string and numeric values
        if isinstance(troponin_value, str) and troponin_value.strip():
            troponin = float(troponin_value)
        elif isinstance(troponin_value, (int, float)):
            troponin = float(troponin_value)
        else:
            troponin = 0

        if troponin > 0.1:  # High troponin
            # Lower thresholds to increase risk
            total_adjustment_factor *= 0.8
        elif troponin > 0.03:  # Elevated troponin
            # Slightly lower thresholds
            total_adjustment_factor *= 0.9
    except (ValueError, TypeError):
        # If troponin value is invalid, skip adjustment
        pass

    # CRP adjustment
    try:
        crp_value = biomarkers.get('crp', '')
        # Handle both string and numeric values
        if isinstance(crp_value, str) and crp_value.strip():
            crp = float(crp_value)
        elif isinstance(crp_value, (int, float)):
            crp = float(crp_value)
        else:
            crp = 0

        if crp > 10:  # High CRP
            # Lower thresholds to increase risk
            total_adjustment_factor *= 0.9
        elif crp > 3:  # Elevated CRP
            # Slightly lower thresholds
            total_adjustment_factor *= 0.95
    except (ValueError, TypeError):
        # If CRP value is invalid, skip adjustment
        pass

    # Apply the total adjustment factor
    adjusted_thresholds['low_medium'] *= total_adjustment_factor
    adjusted_thresholds['medium_high'] *= total_adjustment_factor

    # Ensure thresholds don't go too low
    adjusted_thresholds['low_medium'] = max(0.1, adjusted_thresholds['low_medium'])
    adjusted_thresholds['medium_high'] = max(0.25, adjusted_thresholds['medium_high'])

    return adjusted_thresholds

def get_risk_category(prediction, patient_data):
    """
    Get risk category based on prediction and patient data

    Parameters:
    -----------
    prediction : float
        Prediction score (0-1)
    patient_data : dict
        Dictionary containing patient data

    Returns:
    --------
    str
        Risk category ('Low', 'Medium', or 'High')
    """
    # Extract patient age and gender
    try:
        age = float(patient_data.get('age', 60))
    except (ValueError, TypeError):
        age = 60
    gender = patient_data.get('gender', 'Unknown')

    # Get age and gender adjusted thresholds
    thresholds = get_age_adjusted_thresholds(age, gender)

    # Adjust thresholds based on biomarkers
    biomarkers = patient_data.get('biomarkers', {})
    thresholds = get_biomarker_adjusted_thresholds(thresholds, biomarkers)

    # Determine risk category
    if prediction < thresholds['low_medium']:
        return 'Low'
    elif prediction < thresholds['medium_high']:
        return 'Medium'
    else:
        return 'High'

def get_risk_score_explanation(prediction, patient_data):
    """
    Get explanation of risk score calculation

    Parameters:
    -----------
    prediction : float
        Prediction score (0-1)
    patient_data : dict
        Dictionary containing patient data

    Returns:
    --------
    dict
        Dictionary containing risk score explanation
    """
    # Extract patient age and gender
    try:
        age = float(patient_data.get('age', 60))
    except (ValueError, TypeError):
        age = 60
    gender = patient_data.get('gender', 'Unknown')

    # Get base thresholds
    base_thresholds = get_risk_thresholds()

    # Get age and gender adjusted thresholds
    age_gender_thresholds = get_age_adjusted_thresholds(age, gender)

    # Adjust thresholds based on biomarkers
    biomarkers = patient_data.get('biomarkers', {})
    final_thresholds = get_biomarker_adjusted_thresholds(age_gender_thresholds, biomarkers)

    # Determine risk category
    risk_category = get_risk_category(prediction, patient_data)

    # Create explanation
    explanation = {
        'prediction': float(prediction),
        'risk_category': risk_category,
        'thresholds': {
            'base': base_thresholds,
            'age_gender_adjusted': age_gender_thresholds,
            'final': final_thresholds
        },
        'adjustments': {
            'age': age,
            'gender': gender,
            'biomarkers': biomarkers
        }
    }

    return explanation

def calibrate_risk_thresholds(patient_data_list):
    """
    Calibrate risk thresholds based on patient data

    Parameters:
    -----------
    patient_data_list : list
        List of patient data dictionaries

    Returns:
    --------
    dict
        Dictionary containing calibration results
    """
    if not patient_data_list or len(patient_data_list) < 10:
        return {
            'success': False,
            'message': 'Insufficient data for calibration',
            'thresholds': get_risk_thresholds()
        }

    # Extract predictions
    predictions = []
    for patient in patient_data_list:
        if 'prediction' in patient:
            try:
                prediction = float(patient['prediction'])
                predictions.append(prediction)
            except (ValueError, TypeError):
                continue

    if len(predictions) < 10:
        return {
            'success': False,
            'message': 'Insufficient prediction data for calibration',
            'thresholds': get_risk_thresholds()
        }

    # Sort predictions
    predictions.sort()

    # Calculate percentiles
    low_medium_percentile = 33  # 33rd percentile
    medium_high_percentile = 66  # 66th percentile

    # Calculate thresholds
    low_medium_index = int(len(predictions) * low_medium_percentile / 100)
    medium_high_index = int(len(predictions) * medium_high_percentile / 100)

    low_medium_threshold = predictions[low_medium_index]
    medium_high_threshold = predictions[medium_high_index]

    # Ensure thresholds are reasonable
    low_medium_threshold = max(0.1, min(0.3, low_medium_threshold))
    medium_high_threshold = max(0.3, min(0.6, medium_high_threshold))

    # Create new thresholds
    new_thresholds = {
        'low_medium': low_medium_threshold,
        'medium_high': medium_high_threshold
    }

    # Save new thresholds
    save_risk_thresholds(new_thresholds)

    return {
        'success': True,
        'message': 'Risk thresholds calibrated successfully',
        'thresholds': new_thresholds,
        'num_records': len(predictions),
        'percentiles': {
            'low_medium': low_medium_percentile,
            'medium_high': medium_high_percentile
        }
    }

def register_risk_calibration_routes(app):
    """
    Register risk calibration routes with the Flask app

    Parameters:
    -----------
    app : Flask
        Flask application instance
    """
    from flask import jsonify, request

    @app.route('/api/risk-calibration/thresholds', methods=['GET'])
    def get_risk_calibration_thresholds():
        """
        Endpoint to get risk calibration thresholds
        """
        thresholds = get_risk_thresholds()
        return jsonify(thresholds)

    @app.route('/api/risk-calibration/calibrate', methods=['POST'])
    def calibrate_risk_thresholds_endpoint():
        """
        Endpoint to calibrate risk thresholds
        """
        try:
            # Load patient data
            patient_files = [f for f in os.listdir('data/patients') if f.endswith('.json')]

            if len(patient_files) < 10:
                return jsonify({
                    'success': False,
                    'message': f'Insufficient data for calibration. Need at least 10 records, but only found {len(patient_files)}.',
                    'thresholds': get_risk_thresholds()
                }), 400

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

            # Calibrate thresholds
            result = calibrate_risk_thresholds(patient_data_list)

            # Add timestamp
            result['timestamp'] = datetime.now().isoformat()

            return jsonify(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error calibrating risk thresholds: {str(e)}',
                'thresholds': get_risk_thresholds()
            }), 500

    print("Risk calibration routes registered successfully")
