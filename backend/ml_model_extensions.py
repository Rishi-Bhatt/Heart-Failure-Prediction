"""
ML Model Extensions for Heart Failure Prediction System

This module provides extensions to the existing ML model without modifying the core functionality.
It focuses on implementing Random Forest as an alternative model for heart failure prediction.
"""
import os
import json
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

# Import existing functionality to ensure compatibility
import clinical_ml_model
from clinical_ml_model import engineer_clinical_features

# Define paths for model data
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
RF_MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_model.json')
RF_TRAINING_HISTORY_FILE = os.path.join(MODEL_DIR, 'rf_training_history.json')

class RandomForestHeartFailureModel:
    """
    Random Forest model for heart failure prediction that works alongside
    the existing clinical logistic regression model.
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 class_weight='balanced', random_state=42):
        """
        Initialize the Random Forest model with specified hyperparameters.

        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of the trees
        min_samples_split : int
            Minimum samples required to split a node
        class_weight : str or dict
            Class weights for imbalanced datasets
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state
        )
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}
        self.hyperparams = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'class_weight': class_weight,
            'random_state': random_state
        }

    def fit(self, X, y, feature_names=None):
        """
        Fit the Random Forest model to the training data.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
        """
        # Print the actual number of records being used
        print(f"Random Forest Model: Training with {len(X)} patient records")

        if len(X) < 5:
            print("Warning: Very small training set. Model may not be reliable.")
            return self

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for internal validation if enough samples
        if len(X) >= 10:  # Require more samples for RF validation
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

        self.is_trained = True
        return self

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

        # Feature importance
        metrics['feature_importance'] = dict(zip(self.feature_names, self.model.feature_importances_))

        self.training_metrics = metrics

        print(f"Random Forest model evaluation metrics: {metrics}")

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

        # Get feature importances
        importances = self.model.feature_importances_

        # Calculate standard deviation of feature importances across trees
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

        # Create sorted importance list
        importance_data = []
        for i, feature in enumerate(self.feature_names):
            importance_data.append({
                'feature': feature,
                'importance': float(importances[i]),
                'std': float(std[i]),
                'lower_ci': float(max(0, importances[i] - 1.96 * std[i])),
                'upper_ci': float(min(1, importances[i] + 1.96 * std[i]))
            })

        # Sort by importance
        importance_data.sort(key=lambda x: x['importance'], reverse=True)

        return importance_data

    def explain_prediction(self, features):
        """
        Generate a detailed explanation of the prediction using feature importances
        """
        if not self.is_trained:
            return {
                'probability': 0.5,
                'contributions': {},
                'top_factors': [],
                'feature_importances': {}
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

        # Get feature importances
        importances = self.model.feature_importances_

        # Calculate contribution of each feature
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            # Original feature value
            value = X[0, i]

            # Feature importance
            importance = importances[i]

            # Contribution (simplified approach - multiply value by importance)
            # For a more accurate approach, we would use SHAP values
            contribution = value * importance

            contributions[feature] = {
                'value': float(value),
                'importance': float(importance),
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

        # Generate explanation
        explanation = {
            'probability': float(probability),
            'contributions': contributions,
            'top_factors': top_factors,
            'feature_importances': dict(zip(self.feature_names, importances.tolist()))
        }

        return explanation

    def save_model(self):
        """
        Save the model to file
        """
        if not self.is_trained:
            print("Warning: Attempting to save untrained model")
            return False

        # We can't directly save the sklearn model with json, so we save its parameters
        # and other necessary information to reconstruct it
        model_data = {
            'hyperparams': self.hyperparams,
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat(),
            'feature_importances': self.model.feature_importances_.tolist() if self.is_trained else None
        }

        try:
            # Save model data to JSON
            with open(RF_MODEL_FILE, 'w') as f:
                json.dump(model_data, f, indent=2)

            # Save the actual model using pickle (via joblib)
            joblib.dump(self.model, RF_MODEL_FILE + '.pkl')

            print(f"Model saved to {RF_MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self):
        """
        Load the model from file
        """
        if not os.path.exists(RF_MODEL_FILE):
            print(f"Model file {RF_MODEL_FILE} not found")
            return False

        try:
            # Load model data from JSON
            with open(RF_MODEL_FILE, 'r') as f:
                model_data = json.load(f)

            # Set feature names
            self.feature_names = model_data['feature_names']

            # Set hyperparameters
            self.hyperparams = model_data['hyperparams']

            # Set scaler parameters
            self.scaler.mean_ = np.array(model_data['scaler_mean'])
            self.scaler.scale_ = np.array(model_data['scaler_scale'])

            # Set training metrics
            self.training_metrics = model_data['training_metrics']

            # Load the actual model using pickle (via joblib)

            if os.path.exists(RF_MODEL_FILE + '.pkl'):
                self.model = joblib.load(RF_MODEL_FILE + '.pkl')
            else:
                # If pickle file doesn't exist, recreate the model with saved hyperparameters
                # This won't have the exact same trees, but will have the same hyperparameters
                self.model = RandomForestClassifier(
                    n_estimators=self.hyperparams.get('n_estimators', 100),
                    max_depth=self.hyperparams.get('max_depth', None),
                    min_samples_split=self.hyperparams.get('min_samples_split', 2),
                    class_weight=self.hyperparams.get('class_weight', 'balanced'),
                    random_state=self.hyperparams.get('random_state', 42)
                )
                print("Warning: Model weights not found, recreated model with same hyperparameters")

            self.is_trained = True
            print(f"Model loaded from {RF_MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


# Create a singleton instance for global use
rf_model = RandomForestHeartFailureModel()


def train_random_forest_model(patient_data_list):
    """
    Train the Random Forest model with patient data

    Parameters:
    -----------
    patient_data_list : list
        List of patient data dictionaries

    Returns:
    --------
    dict
        Training results
    """
    # Print the actual number of records available
    num_records = len(patient_data_list) if patient_data_list else 0
    print(f"Random Forest Model: Training with {num_records} patient records")

    if not patient_data_list or len(patient_data_list) < 5:
        print("Insufficient data for Random Forest model training (need at least 5 records)")
        return {
            'success': False,
            'message': f"Insufficient data for Random Forest model training (need at least 5 records, got {num_records})",
            'num_records': num_records
        }

    # Extract features and labels using the same function as the existing ML model
    # This ensures compatibility with the existing data structures
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

        # Extract features using the same function as the existing ML model
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
        print(f"Insufficient labeled data for Random Forest model training: only {len(X)} usable records out of {len(patient_data_list)} total")
        return {
            'success': False,
            'message': f"Insufficient labeled data for Random Forest model training: only {len(X)} usable records out of {len(patient_data_list)} total",
            'num_records': len(X),
            'total_records': len(patient_data_list),
            'processed_count': processed_count,
            'skipped_count': skipped_count
        }

    # Create and train the model
    global rf_model
    rf_model = RandomForestHeartFailureModel()
    rf_model.fit(np.array(X), np.array(y), feature_names=feature_names)

    # Save the model
    rf_model.save_model()

    # Record training event
    training_event = {
        'timestamp': datetime.now().isoformat(),
        'num_records': len(X),
        'total_records': len(patient_data_list),
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'metrics': rf_model.training_metrics,
        'feature_importance': rf_model.get_feature_importance(),
        'message': f"Random Forest model trained successfully with {len(X)} usable records out of {len(patient_data_list)} total"
    }

    # Save training history
    save_training_event(training_event)

    return {
        'success': True,
        'message': f"Random Forest model trained successfully with {len(X)} usable records out of {len(patient_data_list)} total",
        'num_records': len(X),
        'total_records': len(patient_data_list),
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'metrics': rf_model.training_metrics
    }


def save_training_event(training_event):
    """
    Save a training event to the training history
    """
    # Load existing history
    history = []
    if os.path.exists(RF_TRAINING_HISTORY_FILE):
        try:
            with open(RF_TRAINING_HISTORY_FILE, 'r') as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = [history]
        except Exception as e:
            print(f"Error loading training history: {str(e)}")

    # Add new event
    history.append(training_event)

    # Save updated history
    try:
        with open(RF_TRAINING_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved training event with {training_event['num_records']} records")
        return True
    except Exception as e:
        print(f"Error saving training event: {str(e)}")
        return False


def get_training_history():
    """
    Get the Random Forest model training history
    """
    if os.path.exists(RF_TRAINING_HISTORY_FILE):
        try:
            with open(RF_TRAINING_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading RF training history: {str(e)}")

    return []


def predict_heart_failure(patient_data):
    """
    Predict heart failure using the Random Forest model
    """
    # Extract features using the same function as the existing ML model
    features = engineer_clinical_features(patient_data)
    feature_vector = np.array([[v for v in features.values()]])

    # Load model if not already loaded
    global rf_model
    if not rf_model.is_trained:
        if not rf_model.load_model():
            # Fall back to default values if model not available
            return 0.5, 0.7, {
                'probability': 0.5,
                'contributions': {},
                'top_factors': [],
                'feature_importances': {}
            }

    # Make prediction
    probability = rf_model.predict_proba(feature_vector)[0, 1]

    # Get explanation
    explanation = rf_model.explain_prediction(features)

    # Calculate confidence based on model metrics
    # For Random Forest, we can use the variance of predictions across trees as a measure of uncertainty
    # Higher variance = lower confidence
    tree_predictions = np.array([tree.predict_proba(feature_vector)[0, 1] for tree in rf_model.model.estimators_])
    prediction_variance = np.var(tree_predictions)

    # Convert variance to confidence (inverse relationship)
    # Scale to reasonable range (0.7-0.95)
    confidence = 0.95 - min(0.25, prediction_variance * 10)

    return probability, confidence, explanation


def compare_models(patient_data):
    """
    Compare predictions from Random Forest and Logistic Regression models
    """
    # Get prediction from Random Forest model
    rf_probability, rf_confidence, rf_explanation = predict_heart_failure(patient_data)

    # Get prediction from Logistic Regression model (existing model)
    lr_probability, lr_confidence, lr_explanation = clinical_ml_model.predict_heart_failure(patient_data)

    # Compare the predictions
    comparison = {
        'random_forest': {
            'probability': rf_probability,
            'confidence': rf_confidence,
            'explanation': rf_explanation
        },
        'logistic_regression': {
            'probability': lr_probability,
            'confidence': lr_confidence,
            'explanation': lr_explanation
        },
        'agreement': 1.0 - abs(rf_probability - lr_probability),
        'recommendation': 'random_forest' if rf_confidence > lr_confidence else 'logistic_regression'
    }

    return comparison