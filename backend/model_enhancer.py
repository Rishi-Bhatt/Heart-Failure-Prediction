"""
Model Enhancer for Heart Failure Prediction System
This module enhances the existing rule-based model with adaptive weights and feedback learning.
"""
import os
import json
import math
import random
from datetime import datetime

# Define paths for model data
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
WEIGHTS_FILE = os.path.join(MODEL_DIR, 'feature_weights.json')
FEEDBACK_FILE = os.path.join(MODEL_DIR, 'feedback_history.json')
TRAINING_HISTORY_FILE = os.path.join(MODEL_DIR, 'training_history.json')

# Default feature weights
DEFAULT_WEIGHTS = {
    'age': 0.14,
    'gender': 0.05,
    'blood_pressure': 0.09,
    'cholesterol': 0.09,
    'fasting_blood_sugar': 0.05,
    'max_heart_rate': 0.14,
    'exercise_induced_angina': 0.09,
    'st_depression': 0.09,
    'slope_of_st': 0.05,
    'number_of_major_vessels': 0.09,
    'thalassemia': 0.05,
    'prior_cardiac_event': 0.12,  # Slightly reduced from 0.15
    'nt_probnp': 0.05  # New biomarker with conservative initial weight
}

def load_feature_weights():
    """
    Load feature weights from file or use defaults
    """
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                weights = json.load(f)
            print(f"Loaded feature weights from {WEIGHTS_FILE}")
            return weights
        except Exception as e:
            print(f"Error loading weights: {str(e)}")

    # Use default weights if file doesn't exist or has an error
    print("Using default feature weights")
    return DEFAULT_WEIGHTS

def save_feature_weights(weights):
    """
    Save feature weights to file
    """
    try:
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(weights, f, indent=2)
        print(f"Saved feature weights to {WEIGHTS_FILE}")
        return True
    except Exception as e:
        print(f"Error saving weights: {str(e)}")
        return False

def normalize_weights(weights):
    """
    Normalize weights to sum to 1.0
    """
    total = sum(weights.values())
    if total == 0:
        # If all weights are 0, use equal weights
        return {k: 1.0/len(weights) for k in weights}

    return {k: v/total for k, v in weights.items()}

def extract_feature_values(patient_data):
    """
    Extract feature values from patient data
    """
    features = {}

    # Age (normalized to 0-1 range, assuming max age of 100)
    try:
        features['age'] = min(1.0, float(patient_data.get('age', 60)) / 100.0)
    except (ValueError, TypeError):
        features['age'] = 0.6  # Default: 60 years

    # Gender (binary: 1 for male, 0 for female)
    gender = patient_data.get('gender', 'Male')
    features['gender'] = 1.0 if gender == 'Male' else 0.0

    # Blood pressure (normalized to 0-1 range, assuming max of 200)
    try:
        bp = patient_data.get('blood_pressure', '120/80')
        systolic = int(bp.split('/')[0])
        features['blood_pressure'] = min(1.0, systolic / 200.0)
    except (ValueError, TypeError, IndexError):
        features['blood_pressure'] = 0.6  # Default: 120/200

    # Cholesterol (normalized to 0-1 range, assuming max of 300)
    try:
        features['cholesterol'] = min(1.0, float(patient_data.get('cholesterol', 200)) / 300.0)
    except (ValueError, TypeError):
        features['cholesterol'] = 0.67  # Default: 200/300

    # Fasting blood sugar (binary: 1 if > 120, 0 otherwise)
    try:
        fbs = float(patient_data.get('fasting_blood_sugar', 100))
        features['fasting_blood_sugar'] = 1.0 if fbs > 120 else 0.0
    except (ValueError, TypeError):
        features['fasting_blood_sugar'] = 0.0  # Default: below 120

    # Max heart rate (normalized to 0-1 range, assuming max of 220)
    try:
        features['max_heart_rate'] = min(1.0, float(patient_data.get('max_heart_rate', 75)) / 220.0)
    except (ValueError, TypeError):
        features['max_heart_rate'] = 0.34  # Default: 75/220

    # Exercise induced angina (binary: 1 for yes, 0 for no)
    angina = patient_data.get('exercise_induced_angina', False)
    features['exercise_induced_angina'] = 1.0 if angina and angina not in [False, 'false', 'False', '0', 0, None, ''] else 0.0

    # ST depression (normalized to 0-1 range, assuming max of 6.0)
    try:
        features['st_depression'] = min(1.0, float(patient_data.get('st_depression', 0)) / 6.0)
    except (ValueError, TypeError):
        features['st_depression'] = 0.0  # Default: 0

    # Slope of ST segment (0: Upsloping, 0.5: Flat, 1.0: Downsloping)
    slope = patient_data.get('slope_of_st', 'Flat')
    if slope == 'Upsloping':
        features['slope_of_st'] = 0.0
    elif slope == 'Flat':
        features['slope_of_st'] = 0.5
    elif slope == 'Downsloping':
        features['slope_of_st'] = 1.0
    else:
        features['slope_of_st'] = 0.5  # Default: Flat

    # Number of major vessels (normalized to 0-1 range, assuming max of 4)
    try:
        features['number_of_major_vessels'] = min(1.0, float(patient_data.get('number_of_major_vessels', 0)) / 4.0)
    except (ValueError, TypeError):
        features['number_of_major_vessels'] = 0.0  # Default: 0

    # Thalassemia (0: Normal, 0.5: Fixed Defect, 1.0: Reversible Defect)
    thal = patient_data.get('thalassemia', 'Normal')
    if thal == 'Normal':
        features['thalassemia'] = 0.0
    elif thal == 'Fixed Defect':
        features['thalassemia'] = 0.5
    elif thal == 'Reversible Defect':
        features['thalassemia'] = 1.0
    else:
        features['thalassemia'] = 0.0  # Default: Normal

    # Prior cardiac event (0: None, 0.5-1.0: Based on type, severity, and recency)
    prior_event = patient_data.get('prior_cardiac_event', {})
    prior_event_type = prior_event.get('type', '')

    if not prior_event_type:
        features['prior_cardiac_event'] = 0.0  # No prior event
    else:
        # Base risk by event type
        event_risk = {
            'Myocardial Infarction': 1.0,  # Heart attack - highest risk
            'Heart Failure': 0.9,
            'Coronary Artery Disease': 0.8,
            'Arrhythmia': 0.7,
            'Valve Disease': 0.7,
            'Cardiomyopathy': 0.8,
            'Angina': 0.6,
            'Pericarditis': 0.5
        }.get(prior_event_type, 0.5)

        # Adjust for severity
        severity = prior_event.get('severity', 'Mild')
        severity_factor = {
            'Mild': 0.7,
            'Moderate': 0.85,
            'Severe': 1.0
        }.get(severity, 0.7)

        # Adjust for time since event (more recent = higher risk)
        try:
            time_since_event = float(prior_event.get('time_since_event', 12))
            # Exponential decay with time (0-120 months)
            time_factor = math.exp(-0.02 * time_since_event) * 0.8 + 0.2  # Range: 0.2-1.0
        except (ValueError, TypeError):
            time_factor = 0.5  # Default if time is invalid

        # Calculate final risk score for prior event
        features['prior_cardiac_event'] = event_risk * severity_factor * time_factor

    # NT-proBNP biomarker processing based on clinical literature
    # References:
    # 1. Januzzi JL Jr, et al. (2019). "NT-proBNP Testing for Diagnosis and Short-Term Prognosis in Acute Heart Failure"
    # 2. Ponikowski P, et al. (2016). "2016 ESC Guidelines for heart failure"
    try:
        nt_probnp = float(patient_data.get('biomarkers', {}).get('nt_probnp', 0))
        # Age-adjusted thresholds based on ESC Guidelines
        age = float(patient_data.get('age', 65))

        # Age-stratified reference limits (ESC Guidelines 2016)
        if age < 50:
            threshold = 450  # pg/mL
            high_risk = 900  # 2x threshold indicates high risk
        elif age <= 75:
            threshold = 900  # pg/mL
            high_risk = 1800
        else:
            threshold = 1800  # pg/mL
            high_risk = 3600

        # Multi-tier risk stratification based on clinical studies
        if nt_probnp < threshold/2:  # Well below threshold - very low risk
            risk_value = 0.1
        elif nt_probnp < threshold:  # Below threshold but detectable - low risk
            # Linear scaling from 0.1 to 0.5
            risk_value = 0.1 + 0.4 * (nt_probnp - threshold/2) / (threshold/2)
        elif nt_probnp < high_risk:  # Between threshold and high risk - moderate risk
            # Linear scaling from 0.5 to 0.8
            risk_value = 0.5 + 0.3 * (nt_probnp - threshold) / (high_risk - threshold)
        else:  # Above high risk threshold - high risk
            # Logarithmic scaling for very high values to prevent saturation
            # Cap at 0.95 to avoid certainty
            risk_value = min(0.95, 0.8 + 0.15 * math.log(nt_probnp / high_risk + 1))

        features['nt_probnp'] = risk_value

        # Log the risk calculation for research purposes
        print(f"NT-proBNP: {nt_probnp} pg/mL, Age: {age}, Threshold: {threshold}, Risk value: {risk_value:.2f}")
    except (ValueError, TypeError):
        features['nt_probnp'] = 0.0  # Default if missing or invalid

    return features

def calculate_weighted_risk(features, weights):
    """
    Calculate weighted risk score based on features and weights
    """
    risk_score = 0.0

    # Apply weights to each feature
    for feature, value in features.items():
        if feature in weights:
            risk_score += value * weights[feature]

    # Add a small random factor for variability (±2%)
    risk_score += random.uniform(-0.02, 0.02)

    # Ensure risk score is between 0 and 1
    risk_score = max(0.01, min(0.99, risk_score))

    return risk_score

def calculate_confidence(risk_score):
    """
    Calculate confidence based on risk score
    Higher confidence for extreme values, lower for middle range
    """
    # Distance from decision boundary (0.5)
    distance = abs(risk_score - 0.5)

    # Scale to 0-1 range and adjust
    confidence = 0.7 + (distance * 0.6)

    # Ensure reasonable bounds
    confidence = max(0.7, min(0.95, confidence))

    return confidence

def predict_heart_failure(patient_data):
    """
    Predict heart failure risk using enhanced model
    """
    # Load feature weights
    weights = load_feature_weights()

    # Extract feature values
    features = extract_feature_values(patient_data)

    # Calculate risk score
    risk_score = calculate_weighted_risk(features, weights)

    # Calculate confidence
    confidence = calculate_confidence(risk_score)

    # Generate SHAP-like values for explainability
    shap_values = generate_shap_values(features, weights, risk_score)

    return risk_score, confidence, shap_values

def generate_shap_values(features, weights, prediction):
    """
    Generate SHAP-like values for model explainability
    """
    # Base value (average prediction)
    base_value = 0.5

    # Calculate difference from base value
    diff = prediction - base_value

    # Normalize weights
    norm_weights = normalize_weights(weights)

    # Calculate contribution of each feature
    contributions = {}
    for feature, weight in norm_weights.items():
        # Scale by feature value and weight
        if feature in features:
            # Adjust direction based on whether feature increases or decreases risk
            direction = 1.0
            if feature in ['max_heart_rate']:  # Features that decrease risk when high
                direction = -1.0 if features[feature] > 0.5 else 1.0
            elif feature == 'prior_cardiac_event' and features[feature] > 0:  # Prior events always increase risk
                direction = 1.5  # Emphasize the impact of prior events
            elif feature == 'nt_probnp' and features[feature] > 0.5:  # Elevated NT-proBNP increases risk
                direction = 1.6  # Strongly emphasize elevated NT-proBNP

            contributions[feature] = diff * weight * direction

    # Format for frontend
    return {
        'base_value': base_value,
        'values': list(contributions.values()),
        'feature_names': list(contributions.keys())
    }

def record_feedback(patient_id, prediction, actual, patient_data):
    """
    Record feedback for a prediction to use in model improvement
    """
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'patient_id': patient_id,
        'prediction': prediction,
        'actual': actual,
        'features': extract_feature_values(patient_data)
    }

    # Load existing feedback
    feedback_history = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                feedback_history = json.load(f)
            if not isinstance(feedback_history, list):
                feedback_history = [feedback_history]
        except Exception as e:
            print(f"Error loading feedback history: {str(e)}")

    # Add new feedback
    feedback_history.append(feedback_entry)

    # Save updated feedback
    try:
        with open(FEEDBACK_FILE, 'w') as f:
            json.dump(feedback_history, f, indent=2)
        print(f"Saved feedback for patient {patient_id}")
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False

def adjust_weights_from_feedback():
    """
    Adjust feature weights based on feedback history
    """
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback data available for weight adjustment")
        return False

    try:
        # Load feedback history
        with open(FEEDBACK_FILE, 'r') as f:
            feedback_history = json.load(f)

        if not feedback_history or len(feedback_history) < 5:
            print("Insufficient feedback data for weight adjustment")
            return False

        # Load current weights
        weights = load_feature_weights()

        # Calculate weight adjustments
        adjustments = {feature: 0.0 for feature in weights}

        for entry in feedback_history:
            prediction = entry.get('prediction', 0.5)
            actual = entry.get('actual', 0.5)
            features = entry.get('features', {})

            # Calculate error
            error = actual - prediction

            # Adjust weights based on error and feature values
            for feature, value in features.items():
                if feature in adjustments:
                    # Positive error means prediction was too low
                    # Increase weights for features with high values
                    # Decrease weights for features with low values
                    adjustments[feature] += error * value * 0.1

        # Apply adjustments
        for feature, adjustment in adjustments.items():
            weights[feature] += adjustment

        # Ensure all weights are positive
        weights = {k: max(0.01, v) for k, v in weights.items()}

        # Normalize weights
        weights = normalize_weights(weights)

        # Save updated weights
        save_feature_weights(weights)

        # Record training event
        record_training_event(len(feedback_history), weights)

        return True

    except Exception as e:
        print(f"Error adjusting weights: {str(e)}")
        return False

def record_training_event(num_records, weights):
    """
    Record a training event in the training history
    """
    training_event = {
        'timestamp': datetime.now().isoformat(),
        'num_records': num_records,
        'weights': weights,
        'message': f"Model retrained successfully with {num_records} records"
    }

    # Load existing history
    training_history = []
    if os.path.exists(TRAINING_HISTORY_FILE):
        try:
            with open(TRAINING_HISTORY_FILE, 'r') as f:
                training_history = json.load(f)
            if not isinstance(training_history, list):
                training_history = [training_history]
        except Exception as e:
            print(f"Error loading training history: {str(e)}")

    # Add new event
    training_history.append(training_event)

    # Save updated history
    try:
        with open(TRAINING_HISTORY_FILE, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Saved training event with {num_records} records")
        return True
    except Exception as e:
        print(f"Error saving training event: {str(e)}")
        return False

def get_training_history():
    """
    Get the training history
    """
    if os.path.exists(TRAINING_HISTORY_FILE):
        try:
            with open(TRAINING_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading training history: {str(e)}")

    return []

def retrain_model():
    """
    Retrain the model using feedback data
    """
    success = adjust_weights_from_feedback()

    if success:
        # Get the latest training event
        history = get_training_history()
        latest = history[-1] if history else None

        if latest:
            return {
                'success': True,
                'num_records': latest.get('num_records', 0),
                'weights': latest.get('weights', {}),
                'message': latest.get('message', "Model retrained successfully")
            }

    return {
        'success': False,
        'num_records': 0,
        'weights': {},
        'message': "Insufficient data for retraining"
    }

# Initialize weights on module load
if not os.path.exists(WEIGHTS_FILE):
    print("Initializing default feature weights")
    save_feature_weights(DEFAULT_WEIGHTS)
