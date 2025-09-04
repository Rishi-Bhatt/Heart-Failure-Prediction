"""
Clinically-Informed Logistic Regression Model for Heart Failure Prediction
This module implements a logistic regression model that incorporates clinical domain knowledge
through informed priors and feature engineering.
"""
import os
import json
import numpy as np
import math
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

# Define paths for model data
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
ML_MODEL_FILE = os.path.join(MODEL_DIR, 'clinical_lr_model.json')
ML_MODEL_PKL = os.path.join(MODEL_DIR, 'clinical_lr_model.pkl')
ML_TRAINING_HISTORY_FILE = os.path.join(MODEL_DIR, 'ml_training_history.json')

# Define clinical priors based on literature and expert knowledge
# Format: {feature_name: (expected_coefficient, uncertainty)}
# References:
# 1. Januzzi JL Jr, et al. (2019). "NT-proBNP Testing for Diagnosis and Short-Term Prognosis in Acute Heart Failure"
# 2. Ponikowski P, et al. (2016). "2016 ESC Guidelines for heart failure"
# 3. Levy WC, et al. (2006). "The Seattle Heart Failure Model: prediction of survival in heart failure"
# 4. Rahimi K, et al. (2014). "Risk prediction in patients with heart failure: a systematic review and analysis"
CLINICAL_PRIORS = {
    # Demographic factors
    'age': (0.04, 0.01),              # Age increases risk (per year)
    'gender_male': (0.3, 0.1),        # Male gender increases risk

    # Traditional cardiovascular risk factors
    'systolic_bp': (0.02, 0.005),     # Higher systolic BP increases risk (per mmHg)
    'cholesterol': (0.01, 0.003),     # Higher cholesterol increases risk (per mg/dL)
    'fasting_blood_sugar': (0.3, 0.1), # High blood sugar increases risk (>120 mg/dL)

    # Cardiac function indicators
    'max_heart_rate': (-0.01, 0.003), # Lower max heart rate increases risk (per bpm)
    'exercise_angina': (0.5, 0.15),   # Exercise-induced angina increases risk

    # ECG parameters
    'st_depression': (0.2, 0.05),     # ST depression increases risk (per mm)
    'st_slope_flat': (0.4, 0.1),      # Flat ST slope increases risk
    'st_slope_downsloping': (0.6, 0.15), # Downsloping ST increases risk

    # Anatomical factors
    'num_vessels': (0.5, 0.1),        # More vessels affected increases risk (per vessel)
    'thalassemia_reversible': (0.5, 0.15), # Reversible defect increases risk

    # Biomarkers - based on clinical studies
    'nt_probnp': (0.8, 0.15),         # NT-proBNP is a strong predictor (normalized value)
    'nt_probnp_threshold_ratio': (0.6, 0.1)  # Ratio to age-specific threshold (clinical significance)
}

class ClinicallyInformedLogisticRegression:
    def __init__(self, clinical_priors=None, regularization_strength=1.0):
        """
        Initialize the model with clinical prior knowledge

        Parameters:
        -----------
        clinical_priors : dict
            Dictionary mapping feature names to prior distributions (mean, std)
        regularization_strength : float
            Strength of regularization (inverse of C in sklearn)
        """
        self.clinical_priors = clinical_priors or CLINICAL_PRIORS
        self.model = LogisticRegression(
            C=1.0/regularization_strength,
            penalty='l2',
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced'
        )
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}

    def fit(self, X, y, feature_names=None):
        """
        Fit the model with prior knowledge integration
        """
        # Print the actual number of records being used
        print(f"ML Model: Training with {len(X)} patient records")

        if len(X) < 5:
            print("Warning: Very small training set. Model may not be reliable.")
            return self

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for internal validation if enough samples (use a percentage instead of fixed number)
        # Use 20% for validation if we have at least 5 records (so at least 1 for validation)
        if len(X) >= 5:
            # Calculate test size to ensure at least 1 record for validation but not more than 20%
            test_size = min(0.2, max(1/len(X), 0.05))
            print(f"Using {1-test_size:.1%} of data for training, {test_size:.1%} for validation")

            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42
            )

            # Train the model
            self.model.fit(X_train, y_train)

            # Evaluate on validation set
            self._evaluate_model(X_val, y_val)
        else:
            # Use all data for training if very limited samples
            print("Using all data for training (no validation split)")
            self.model.fit(X_scaled, y)
            self._evaluate_model(X_scaled, y)

        # Adjust coefficients based on clinical priors
        self._adjust_coefficients_with_priors()

        self.is_trained = True
        return self

    def _adjust_coefficients_with_priors(self):
        """
        Adjust model coefficients based on clinical priors
        """
        if not self.feature_names:
            return

        coef = self.model.coef_[0].copy()

        for i, feature in enumerate(self.feature_names):
            if feature in self.clinical_priors:
                prior_mean, prior_std = self.clinical_priors[feature]

                # Bayesian update (simplified)
                # Combine the ML-derived coefficient with the clinical prior
                ml_mean = coef[i]
                ml_std = 1.0  # This would ideally be derived from the data

                # Bayesian combination of two Gaussians
                posterior_precision = 1/prior_std**2 + 1/ml_std**2
                posterior_mean = (prior_mean/prior_std**2 + ml_mean/ml_std**2) / posterior_precision
                posterior_std = math.sqrt(1/posterior_precision)

                # Update coefficient
                coef[i] = posterior_mean

                print(f"Adjusted {feature}: {ml_mean:.4f} -> {posterior_mean:.4f} (prior: {prior_mean:.4f})")

        # Update model coefficients
        self.model.coef_[0] = coef

    def _evaluate_model(self, X, y):
        """
        Evaluate model performance and store metrics
        """
        # Get predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = self.model.predict(X)

        # Calculate metrics
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = np.mean(y_pred == y)

        # ROC AUC
        if len(np.unique(y)) > 1:  # Only calculate if both classes present
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)

            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        else:
            metrics['roc_auc'] = 0.5
            metrics['pr_auc'] = 0.5

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Sensitivity and Specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Positive and Negative Predictive Values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

        self.training_metrics = metrics

        print(f"Model evaluation metrics: {metrics}")

    def predict_proba(self, X):
        """
        Predict probability of heart failure
        """
        if not self.is_trained:
            print("Warning: Model not trained yet. Using default probabilities.")
            return np.array([[0.5, 0.5]] * len(X))

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get probabilities
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        """
        Make binary predictions
        """
        if not self.is_trained:
            print("Warning: Model not trained yet. Using default predictions.")
            return np.array([0] * len(X))

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get predictions
        return self.model.predict(X_scaled)

    def get_feature_importance(self):
        """
        Get feature importance with confidence intervals
        """
        if not self.is_trained or not self.feature_names:
            return None

        # Get coefficients
        coef = self.model.coef_[0]

        # Calculate odds ratios
        odds_ratios = np.exp(coef)

        # For a proper publication, you would calculate proper CIs
        # This is a simplified version
        lower_ci = np.exp(coef - 1.96)  # Simplified
        upper_ci = np.exp(coef + 1.96)  # Simplified

        # Create sorted importance list
        importance_data = []
        for i, feature in enumerate(self.feature_names):
            importance_data.append({
                'feature': feature,
                'coefficient': float(coef[i]),
                'odds_ratio': float(odds_ratios[i]),
                'lower_ci': float(lower_ci[i]),
                'upper_ci': float(upper_ci[i])
            })

        # Sort by absolute coefficient value
        importance_data.sort(key=lambda x: abs(x['coefficient']), reverse=True)

        return importance_data

    def explain_prediction(self, features):
        """
        Generate a detailed explanation of the prediction
        """
        if not self.is_trained:
            return {
                'probability': 0.5,
                'base_value': 0,
                'contributions': {},
                'top_factors': [],
                'clinical_alignment': {}
            }

        # Convert features to array
        if isinstance(features, dict):
            # Create feature vector in the correct order
            feature_vector = []
            for feature in self.feature_names:
                feature_vector.append(features.get(feature, 0))
            X = np.array([feature_vector])
        else:
            X = np.array([features])

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Get prediction
        probability = self.model.predict_proba(X_scaled)[0, 1]

        # Calculate contribution of each feature
        base_value = self.model.intercept_[0]
        contributions = {}

        for i, feature in enumerate(self.feature_names):
            # Original feature value
            value = X[0, i]

            # Scaled feature value
            scaled_value = X_scaled[0, i]

            # Coefficient
            coef = self.model.coef_[0, i]

            # Contribution
            contribution = scaled_value * coef

            contributions[feature] = {
                'value': float(value),
                'coefficient': float(coef),
                'contribution': float(contribution)
            }

        # Sort contributions by absolute magnitude
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )

        # Get top factors
        top_factors = sorted_contributions[:5]

        # Assess clinical alignment
        clinical_alignment = self._assess_clinical_alignment(contributions)

        # Generate explanation
        explanation = {
            'probability': float(probability),
            'base_value': float(base_value),
            'contributions': contributions,
            'top_factors': top_factors,
            'clinical_alignment': clinical_alignment
        }

        return explanation

    def _assess_clinical_alignment(self, contributions):
        """
        Assess how well the prediction aligns with clinical knowledge
        """
        alignment = {}

        for feature, contribution_data in contributions.items():
            if feature in self.clinical_priors:
                prior_mean, _ = self.clinical_priors[feature]

                # Check if contribution direction matches clinical expectation
                expected_direction = np.sign(prior_mean)
                actual_direction = np.sign(contribution_data['coefficient'])

                alignment[feature] = {
                    'expected_direction': float(expected_direction),
                    'actual_direction': float(actual_direction),
                    'aligned': bool(expected_direction == actual_direction),
                    'contribution': float(contribution_data['contribution'])
                }

        return alignment

    def save_model(self):
        """
        Save the model to file
        """
        if not self.is_trained:
            print("Warning: Attempting to save untrained model")
            return False

        model_data = {
            'coefficients': self.model.coef_[0].tolist(),
            'intercept': float(self.model.intercept_[0]),
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(ML_MODEL_FILE, 'w') as f:
                json.dump(model_data, f, indent=2)
            print(f"Model saved to {ML_MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self):
        """
        Load the model from file
        """
        # First try to load from pickle file if available
        if os.path.exists(ML_MODEL_PKL):
            try:
                import joblib
                self.model = joblib.load(ML_MODEL_PKL)

                # Load JSON file for other parameters
                if os.path.exists(ML_MODEL_FILE):
                    with open(ML_MODEL_FILE, 'r') as f:
                        model_data = json.load(f)

                    # Set feature names
                    self.feature_names = model_data['feature_names']

                    # Set scaler parameters
                    self.scaler.mean_ = np.array(model_data['scaler_mean'])
                    self.scaler.scale_ = np.array(model_data['scaler_scale'])

                    # Set training metrics
                    self.training_metrics = model_data['training_metrics']

                self.is_trained = True
                print(f"Model loaded from {ML_MODEL_PKL}")
                return True
            except Exception as e:
                print(f"Error loading model from pickle: {str(e)}")
                # Fall back to JSON loading

        # Fall back to JSON loading
        if not os.path.exists(ML_MODEL_FILE):
            print(f"Model file {ML_MODEL_FILE} not found")
            return False

        try:
            with open(ML_MODEL_FILE, 'r') as f:
                model_data = json.load(f)

            # Set feature names
            self.feature_names = model_data['feature_names']

            # Set coefficients and intercept
            self.model.coef_ = np.array([model_data['coefficients']])
            self.model.intercept_ = np.array([model_data['intercept']])

            # Set scaler parameters
            self.scaler.mean_ = np.array(model_data['scaler_mean'])
            self.scaler.scale_ = np.array(model_data['scaler_scale'])

            # Set training metrics
            self.training_metrics = model_data['training_metrics']

            # Set classes_ attribute (required for predict_proba)
            self.model.classes_ = np.array([0, 1])

            self.is_trained = True
            print(f"Model loaded from {ML_MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Error loading model from JSON: {str(e)}")
            return False

def engineer_clinical_features(patient_data):
    """
    Engineer clinically relevant features from raw patient data
    """
    features = {}

    # Basic demographics
    try:
        features['age'] = float(patient_data.get('age', 60)) / 100.0  # Normalize to 0-1
    except (ValueError, TypeError):
        features['age'] = 0.6  # Default

    # Gender (one-hot encoded)
    gender = patient_data.get('gender', 'Male')
    features['gender_male'] = 1.0 if gender == 'Male' else 0.0

    # Blood pressure - extract systolic
    try:
        bp = patient_data.get('blood_pressure', '120/80')
        features['systolic_bp'] = float(bp.split('/')[0]) / 200.0  # Normalize
    except (ValueError, TypeError, IndexError):
        features['systolic_bp'] = 0.6  # Default (120/200)

    # Cholesterol (normalized)
    try:
        features['cholesterol'] = float(patient_data.get('cholesterol', 200)) / 300.0
    except (ValueError, TypeError):
        features['cholesterol'] = 0.67  # Default

    # Fasting blood sugar > 120 mg/dl
    try:
        fbs = float(patient_data.get('fasting_blood_sugar', 100))
        features['fasting_blood_sugar'] = 1.0 if fbs > 120 else 0.0
    except (ValueError, TypeError):
        features['fasting_blood_sugar'] = 0.0  # Default

    # Max heart rate (normalized and inverted - lower is higher risk)
    try:
        max_hr = float(patient_data.get('max_heart_rate', 150))
        features['max_heart_rate'] = 1.0 - (max_hr / 220.0)  # Normalize and invert
    except (ValueError, TypeError):
        features['max_heart_rate'] = 0.32  # Default

    # Exercise induced angina
    angina = patient_data.get('exercise_induced_angina', False)
    features['exercise_angina'] = 1.0 if angina and angina not in [False, 'false', 'False', '0', 0, None, ''] else 0.0

    # ST depression
    try:
        features['st_depression'] = float(patient_data.get('st_depression', 0)) / 6.0
    except (ValueError, TypeError):
        features['st_depression'] = 0.0  # Default

    # ST slope (one-hot encoded)
    slope = patient_data.get('slope_of_st', 'Flat')
    features['st_slope_flat'] = 1.0 if slope == 'Flat' else 0.0
    features['st_slope_downsloping'] = 1.0 if slope == 'Downsloping' else 0.0

    # Number of major vessels
    try:
        features['num_vessels'] = float(patient_data.get('number_of_major_vessels', 0)) / 4.0
    except (ValueError, TypeError):
        features['num_vessels'] = 0.0  # Default

    # Thalassemia (one-hot encoded)
    thal = patient_data.get('thalassemia', 'Normal')
    features['thalassemia_fixed'] = 1.0 if thal == 'Fixed Defect' else 0.0
    features['thalassemia_reversible'] = 1.0 if thal == 'Reversible Defect' else 0.0

    # NT-proBNP biomarker processing with evidence-based risk stratification
    # References:
    # 1. Januzzi JL Jr, et al. (2019). "NT-proBNP Testing for Diagnosis and Short-Term Prognosis in Acute Heart Failure"
    # 2. Ibrahim NE, Januzzi JL Jr. (2018). "The Future of Biomarker-Guided Therapy for Heart Failure"
    try:
        nt_probnp = float(patient_data.get('biomarkers', {}).get('nt_probnp', 0))
        if nt_probnp > 0:
            # Age-adjusted reference ranges based on clinical guidelines
            age = float(patient_data.get('age', 65))

            # Define age-stratified thresholds (pg/mL)
            if age < 50:
                rule_in_threshold = 450
                rule_out_threshold = 300
            elif age <= 75:
                rule_in_threshold = 900
                rule_out_threshold = 500
            else:
                rule_in_threshold = 1800
                rule_out_threshold = 1000

            # Calculate normalized value using piecewise function based on clinical cutoffs
            if nt_probnp < rule_out_threshold:
                # Below rule-out threshold: very low risk
                # Use log scale for values below rule-out threshold
                if nt_probnp < 50:  # Minimal detectable level
                    normalized_value = 0.05  # Minimal baseline risk
                else:
                    # Log scale from 0.05 to 0.3
                    normalized_value = 0.05 + 0.25 * (math.log(nt_probnp) - math.log(50)) / (math.log(rule_out_threshold) - math.log(50))
            elif nt_probnp < rule_in_threshold:
                # Between rule-out and rule-in: moderate risk
                # Linear scale from 0.3 to 0.7
                normalized_value = 0.3 + 0.4 * (nt_probnp - rule_out_threshold) / (rule_in_threshold - rule_out_threshold)
            else:
                # Above rule-in threshold: high risk
                # Log scale for high values to prevent saturation
                # Cap at 0.98 to avoid certainty
                normalized_value = min(0.98, 0.7 + 0.28 * math.log(nt_probnp / rule_in_threshold + 1) / math.log(5))

            features['nt_probnp'] = normalized_value

            # Add research-relevant derived features
            # NT-proBNP ratio to age-specific threshold (clinical literature shows this is predictive)
            features['nt_probnp_threshold_ratio'] = nt_probnp / rule_in_threshold

            # Log the calculation for research purposes
            print(f"ML Model - NT-proBNP: {nt_probnp} pg/mL, Age: {age}, Normalized value: {normalized_value:.3f}")
        else:
            features['nt_probnp'] = 0.0
            features['nt_probnp_threshold_ratio'] = 0.0
    except (ValueError, TypeError):
        features['nt_probnp'] = 0.0  # Default if missing or invalid
        features['nt_probnp_threshold_ratio'] = 0.0

    # Clinical interaction terms based on cardiovascular pathophysiology and literature
    # References:
    # 1. Levy WC, et al. (2006). "The Seattle Heart Failure Model"
    # 2. Rahimi K, et al. (2014). "Risk prediction in patients with heart failure"

    # Age and systolic BP interaction (higher risk in elderly with high BP)
    # Physiological basis: Reduced arterial compliance in elderly magnifies BP effects
    features['age_systolic_interaction'] = features['age'] * features['systolic_bp']

    # Exercise angina and ST depression interaction
    # Physiological basis: Combination indicates more severe ischemia than either alone
    features['angina_st_interaction'] = features['exercise_angina'] * features['st_depression']

    # NT-proBNP and age interaction
    # Physiological basis: NT-proBNP has different clinical significance by age group
    # Clinical evidence: Age-stratified cutoffs in ESC guidelines
    features['nt_probnp_age_interaction'] = features['nt_probnp'] * features['age']

    # NT-proBNP and ST depression interaction
    # Physiological basis: Combination of myocardial strain (BNP) and ischemia (ST)
    # indicates more severe cardiac pathology
    features['nt_probnp_st_interaction'] = features['nt_probnp'] * features['st_depression']

    # NT-proBNP and number of vessels interaction
    # Physiological basis: Elevated NT-proBNP with multi-vessel disease indicates
    # more extensive cardiac damage and higher risk
    features['nt_probnp_vessels_interaction'] = features['nt_probnp'] * features['num_vessels']

    # Age and max heart rate interaction (chronotropic incompetence in elderly)
    # Physiological basis: Inability to increase heart rate with age indicates
    # autonomic dysfunction and worse prognosis
    if features['max_heart_rate'] < 0.5:  # Lower max heart rate is worse
        features['age_hr_interaction'] = features['age'] * (1 - features['max_heart_rate'])
    else:
        features['age_hr_interaction'] = 0

    # MEDIUM RISK SPECIFIC FEATURES
    # These features are specifically designed to better differentiate medium-risk patients

    # 1. Biomarker Gradient Features - better capture the transition zone between low and high risk
    try:
        nt_probnp = float(patient_data.get('biomarkers', {}).get('nt_probnp', 0))
        # Medium-risk specific NT-proBNP feature (peaks in the medium range)
        # This creates a bell curve that peaks in the medium risk range (300-900 pg/mL)
        if 300 <= nt_probnp <= 900:
            # Normalized to peak at 1.0 in the middle of the range (600 pg/mL)
            distance_from_center = abs(nt_probnp - 600) / 300
            features['medium_risk_nt_probnp'] = 1.0 - distance_from_center
        else:
            features['medium_risk_nt_probnp'] = 0.0

        # Troponin medium-risk feature
        troponin = float(patient_data.get('biomarkers', {}).get('troponin', 0))
        if 0.03 <= troponin <= 0.1:
            # Normalized to peak at 1.0 in the middle of the range (0.065)
            distance_from_center = abs(troponin - 0.065) / 0.035
            features['medium_risk_troponin'] = 1.0 - distance_from_center
        else:
            features['medium_risk_troponin'] = 0.0
    except (ValueError, TypeError):
        features['medium_risk_nt_probnp'] = 0.0
        features['medium_risk_troponin'] = 0.0

    # 2. Age-specific medium risk features
    try:
        age = float(patient_data.get('age', 60))
        # Medium-risk age feature (peaks in middle age range 45-65)
        if 45 <= age <= 65:
            # Normalized to peak at 1.0 in the middle of the range (55)
            distance_from_center = abs(age - 55) / 10
            features['medium_risk_age'] = 1.0 - distance_from_center
        else:
            features['medium_risk_age'] = 0.0
    except (ValueError, TypeError):
        features['medium_risk_age'] = 0.0

    # 3. Blood pressure medium risk feature (peaks in the pre-hypertensive range)
    try:
        bp = patient_data.get('blood_pressure', '120/80')
        systolic = float(bp.split('/')[0])
        # Medium-risk BP feature (peaks in pre-hypertensive range 130-150)
        if 130 <= systolic <= 150:
            # Normalized to peak at 1.0 in the middle of the range (140)
            distance_from_center = abs(systolic - 140) / 10
            features['medium_risk_bp'] = 1.0 - distance_from_center
        else:
            features['medium_risk_bp'] = 0.0
    except (ValueError, TypeError, IndexError):
        features['medium_risk_bp'] = 0.0

    # 4. Cholesterol medium risk feature (peaks in borderline high range)
    try:
        cholesterol = float(patient_data.get('cholesterol', 200))
        # Medium-risk cholesterol feature (peaks in borderline high range 200-240)
        if 200 <= cholesterol <= 240:
            # Normalized to peak at 1.0 in the middle of the range (220)
            distance_from_center = abs(cholesterol - 220) / 20
            features['medium_risk_cholesterol'] = 1.0 - distance_from_center
        else:
            features['medium_risk_cholesterol'] = 0.0
    except (ValueError, TypeError):
        features['medium_risk_cholesterol'] = 0.0

    # 5. Composite medium risk score (combination of medium risk features)
    medium_risk_features = [
        features.get('medium_risk_nt_probnp', 0),
        features.get('medium_risk_troponin', 0),
        features.get('medium_risk_age', 0),
        features.get('medium_risk_bp', 0),
        features.get('medium_risk_cholesterol', 0)
    ]
    features['composite_medium_risk_score'] = sum(medium_risk_features) / len(medium_risk_features)

    # 6. Medium risk interaction terms
    features['medium_risk_bp_cholesterol'] = features.get('medium_risk_bp', 0) * features.get('medium_risk_cholesterol', 0)
    features['medium_risk_age_biomarker'] = features.get('medium_risk_age', 0) * features.get('medium_risk_nt_probnp', 0)

    return features

def train_ml_model(patient_data_list):
    """
    Train the ML model with patient data
    """
    # Print the actual number of records available
    num_records = len(patient_data_list) if patient_data_list else 0
    print(f"Clinical ML Model: Training with {num_records} patient records")

    if not patient_data_list or len(patient_data_list) < 5:
        print("Insufficient data for ML model training (need at least 5 records)")
        return {
            'success': False,
            'message': f"Insufficient data for ML model training (need at least 5 records, got {num_records})",
            'num_records': num_records
        }

    # Extract features and labels
    X = []
    y = []
    feature_names = None
    processed_count = 0
    skipped_count = 0

    print(f"Processing {len(patient_data_list)} patient records for feature extraction")

    for i, patient in enumerate(patient_data_list):
        # Extract patient data
        patient_data = patient.get('patient_data', {})
        if not patient_data:
            print(f"Skipping patient {i+1}: No patient_data found")
            skipped_count += 1
            continue

        # Debug the patient data structure
        print(f"Patient {i+1} data structure: {list(patient.keys())}")
        print(f"Patient {i+1} data fields: {list(patient_data.keys())}")

        # Extract features
        try:
            features = engineer_clinical_features(patient_data)

            if feature_names is None:
                feature_names = list(features.keys())
                print(f"Feature names: {feature_names}")

            # Create feature vector
            feature_vector = [features.get(feature, 0) for feature in feature_names]
            X.append(feature_vector)

            # Extract label (feedback or prediction)
            if 'feedback' in patient and patient['feedback'] is not None:
                # Use feedback as label (1 for correct prediction of heart failure)
                label = 1 if patient['feedback'] == 'correct' else 0
                y.append(label)
                print(f"Patient {i+1} (ID: {patient.get('patient_id', 'unknown')}): Using feedback as label: {label}")
            elif 'prediction' in patient:
                # Use prediction as proxy (not ideal but workable)
                try:
                    prediction_value = float(patient['prediction'])
                    label = 1 if prediction_value >= 0.5 else 0
                    y.append(label)
                    print(f"Patient {i+1} (ID: {patient.get('patient_id', 'unknown')}): Using prediction as label: {label} (from {prediction_value})")
                except (ValueError, TypeError) as e:
                    # Skip this patient if prediction is not a valid number
                    X.pop()  # Remove the feature vector we just added
                    print(f"Skipping patient {i+1} (ID: {patient.get('patient_id', 'unknown')}): Invalid prediction value: {patient['prediction']}")
                    skipped_count += 1
                    continue
            else:
                # Skip this patient if no label available
                X.pop()  # Remove the feature vector we just added
                print(f"Skipping patient {i+1} (ID: {patient.get('patient_id', 'unknown')}): No label available")
                skipped_count += 1
                continue

            processed_count += 1
        except Exception as e:
            print(f"Error processing patient {i+1} (ID: {patient.get('patient_id', 'unknown')}): {str(e)}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue

    print(f"Feature extraction complete: {processed_count} patients processed, {skipped_count} skipped")

    if len(X) < 5:
        print(f"Insufficient labeled data for ML model training: only {len(X)} usable records out of {len(patient_data_list)} total")
        return {
            'success': False,
            'message': f"Insufficient labeled data for ML model training: only {len(X)} usable records out of {len(patient_data_list)} total",
            'num_records': len(X),
            'total_records': len(patient_data_list),
            'processed_count': processed_count,
            'skipped_count': skipped_count
        }

    # Create and train the model
    model = ClinicallyInformedLogisticRegression()
    model.fit(np.array(X), np.array(y), feature_names=feature_names)

    # Save the model
    model.save_model()

    # Record training event
    training_event = {
        'timestamp': datetime.now().isoformat(),
        'num_records': len(X),
        'total_records': len(patient_data_list),
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'metrics': model.training_metrics,
        'feature_importance': model.get_feature_importance(),
        'message': f"ML model trained successfully with {len(X)} usable records out of {len(patient_data_list)} total"
    }

    # Save training history
    save_training_event(training_event)

    return {
        'success': True,
        'message': f"ML model trained successfully with {len(X)} usable records out of {len(patient_data_list)} total",
        'num_records': len(X),
        'total_records': len(patient_data_list),
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'metrics': model.training_metrics
    }

def save_training_event(training_event):
    """
    Save a training event to the training history
    """
    # Load existing history
    history = []
    if os.path.exists(ML_TRAINING_HISTORY_FILE):
        try:
            with open(ML_TRAINING_HISTORY_FILE, 'r') as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = [history]
        except Exception as e:
            print(f"Error loading training history: {str(e)}")

    # Add new event
    history.append(training_event)

    # Save updated history
    try:
        with open(ML_TRAINING_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training event with {training_event['num_records']} records")
        return True
    except Exception as e:
        print(f"Error saving training event: {str(e)}")
        return False

def get_training_history():
    """
    Get the ML model training history
    """
    if os.path.exists(ML_TRAINING_HISTORY_FILE):
        try:
            with open(ML_TRAINING_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading ML training history: {str(e)}")

    return []

def predict_heart_failure(patient_data):
    """
    Predict heart failure using the ML model
    """
    # Extract features
    features = engineer_clinical_features(patient_data)
    feature_vector = np.array([[v for v in features.values()]])

    # Create model and load saved parameters
    model = ClinicallyInformedLogisticRegression()
    if model.load_model():
        # Make prediction
        probability = model.predict_proba(feature_vector)[0, 1]

        # Get explanation
        explanation = model.explain_prediction(features)

        # Calculate confidence based on clinical alignment
        alignment_score = sum(1 for f in explanation['clinical_alignment'].values() if f['aligned']) / max(1, len(explanation['clinical_alignment']))
        confidence = 0.7 + (alignment_score * 0.25)  # Scale to 0.7-0.95

        return probability, confidence, explanation
    else:
        # Return default values if model not available
        return 0.5, 0.7, {
            'probability': 0.5,
            'base_value': 0,
            'contributions': {},
            'top_factors': [],
            'clinical_alignment': {}
        }
