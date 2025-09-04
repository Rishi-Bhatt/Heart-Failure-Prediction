"""
Gradient Boosting Model for Heart Failure Prediction

This module implements a gradient boosting model (XGBoost) for heart failure prediction
with hyperparameter optimization and feature importance analysis.
"""

import os
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import io
import base64

# Define paths for model data
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
GB_MODEL_FILE = os.path.join(MODEL_DIR, 'gradient_boosting_model.json')
GB_MODEL_PKL = os.path.join(MODEL_DIR, 'gradient_boosting_model.pkl')
GB_TRAINING_HISTORY_FILE = os.path.join(MODEL_DIR, 'gb_training_history.json')

class GradientBoostingModel:
    """
    Gradient Boosting model for heart failure prediction that works alongside
    the existing models in the ensemble.
    """
    def __init__(self, learning_rate=0.05, max_depth=6, n_estimators=100,
                 subsample=0.8, colsample_bytree=0.8, random_state=42):
        """
        Initialize the Gradient Boosting model with specified hyperparameters.

        Parameters:
        -----------
        learning_rate : float
            Learning rate for the model
        max_depth : int
            Maximum depth of the trees
        n_estimators : int
            Number of boosting rounds
        subsample : float
            Subsample ratio of the training instances
        colsample_bytree : float
            Subsample ratio of columns when constructing each tree
        random_state : int
            Random seed for reproducibility
        """
        self.hyperparams = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': random_state,
            'use_label_encoder': False,
            'verbosity': 0
        }

        self.model = xgb.XGBClassifier(**self.hyperparams)
        self.feature_names = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_metrics = {}

    def fit(self, X, y, feature_names=None, eval_set=None, early_stopping_rounds=20):
        """
        Fit the Gradient Boosting model to the training data.

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        feature_names : list
            Names of features
        eval_set : list
            Validation set for early stopping
        early_stopping_rounds : int
            Number of rounds with no improvement to stop training

        Returns:
        --------
        self : object
            Returns self
        """
        # Print the actual number of records being used
        print(f"Gradient Boosting Model: Training with {len(X)} patient records")

        if len(X) < 10:
            print("Warning: Very small training set. Model may not be reliable.")
            return self

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data for internal validation if enough samples
        if len(X) >= 20:  # Require more samples for proper validation
            test_size = min(0.2, max(1/len(X), 0.1))
            print(f"Using {1-test_size:.1%} of data for training, {test_size:.1%} for validation")

            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )

            # Create evaluation set for early stopping
            eval_set = [(X_train, y_train), (X_val, y_val)]

            # Train the model with early stopping
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )

            # Evaluate on validation set
            self._evaluate_model(X_val, y_val)
        else:
            # Use all data for training if very limited samples
            print("Using all data for training (no validation split)")
            self.model.fit(X_scaled, y)
            self._evaluate_model(X_scaled, y)

        self.is_trained = True

        # Save the model
        self.save_model()

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
        metrics['accuracy'] = float(accuracy_score(y, y_pred))

        # ROC AUC
        if len(np.unique(y)) > 1:  # Only calculate if both classes present
            metrics['roc_auc'] = float(roc_auc_score(y, y_pred_proba))

            # Precision, Recall, F1
            metrics['precision'] = float(precision_score(y, y_pred, zero_division=0))
            metrics['recall'] = float(recall_score(y, y_pred, zero_division=0))
            metrics['f1'] = float(f1_score(y, y_pred, zero_division=0))
        else:
            metrics['roc_auc'] = 0.5
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
            metrics['f1'] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Sensitivity and Specificity
        metrics['sensitivity'] = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics['specificity'] = float(tn / (tn + fp) if (tn + fp) > 0 else 0)

        # Feature importance - check if feature_names is available
        if hasattr(self, 'feature_names') and self.feature_names is not None and len(self.feature_names) > 0:
            try:
                # Check if model has feature_importances_ attribute
                if hasattr(self.model, 'feature_importances_'):
                    # Make sure feature_names and feature_importances_ have the same length
                    if len(self.feature_names) == len(self.model.feature_importances_):
                        metrics['feature_importance'] = dict(zip(
                            self.feature_names,
                            [float(x) for x in self.model.feature_importances_]
                        ))
                    else:
                        print(f"Warning: Feature names length ({len(self.feature_names)}) doesn't match feature importances length ({len(self.model.feature_importances_)})")
                        metrics['feature_importance'] = {}
                else:
                    print("Warning: Model doesn't have feature_importances_ attribute")
                    metrics['feature_importance'] = {}
            except Exception as e:
                print(f"Error calculating feature importance: {str(e)}")
                metrics['feature_importance'] = {}
        else:
            print("Warning: No feature names available for feature importance calculation")
            metrics['feature_importance'] = {}

        self.training_metrics = metrics

        print(f"Gradient Boosting model evaluation metrics: {metrics}")

    def predict_proba(self, X):
        """
        Predict probability of heart failure
        """
        if not self.is_trained:
            print("Warning: Model not trained yet. Using default probabilities.")
            return np.array([[0.5, 0.5]] * len(X))

        try:
            # Try to use feature_utils for consistent feature handling
            try:
                # If X is a DataFrame, ensure feature consistency
                if hasattr(X, 'columns'):
                    from utils.feature_utils import ensure_feature_consistency
                    X = ensure_feature_consistency(X)
                    print("Using feature_utils for consistent feature handling in GB model")
            except ImportError:
                # Fallback if utils module is not available
                pass

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get probabilities
            return self.model.predict_proba(X_scaled)
        except Exception as e:
            print(f"Error in GB predict_proba: {str(e)}")
            # Fallback to default probabilities
            return np.array([[0.5, 0.5]] * len(X))

    def predict(self, X):
        """
        Make binary predictions
        """
        if not self.is_trained:
            print("Warning: Model not trained yet. Using default predictions.")
            return np.array([0] * len(X))

        try:
            # Try to use feature_utils for consistent feature handling
            try:
                # If X is a DataFrame, ensure feature consistency
                if hasattr(X, 'columns'):
                    from utils.feature_utils import ensure_feature_consistency
                    X = ensure_feature_consistency(X)
                    print("Using feature_utils for consistent feature handling in GB model")
            except ImportError:
                # Fallback if utils module is not available
                pass

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Get predictions
            return self.model.predict(X_scaled)
        except Exception as e:
            print(f"Error in GB predict: {str(e)}")
            # Fallback to default predictions
            return np.array([0] * len(X))

    def get_feature_importance(self):
        """
        Get feature importance with confidence intervals
        """
        if not self.is_trained:
            return None

        # Check if feature_names is available
        if not hasattr(self, 'feature_names') or self.feature_names is None or len(self.feature_names) == 0:
            print("Warning: No feature names available for feature importance calculation")
            return None

        # Check if model has feature_importances_ attribute
        if not hasattr(self.model, 'feature_importances_'):
            print("Warning: Model doesn't have feature_importances_ attribute")
            return None

        try:
            # Get feature importances
            importances = self.model.feature_importances_

            # Check if feature_names and feature_importances_ have the same length
            if len(self.feature_names) != len(importances):
                print(f"Warning: Feature names length ({len(self.feature_names)}) doesn't match feature importances length ({len(importances)})")
                return None

            # Create sorted importance list
            importance_data = []
            for i, feature in enumerate(self.feature_names):
                importance_data.append({
                    'feature': feature,
                    'importance': float(importances[i])
                })

            # Sort by importance
            importance_data.sort(key=lambda x: x['importance'], reverse=True)

            return importance_data
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            return None

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

        # We can't directly save the xgboost model with json, so we save its parameters
        # and other necessary information to reconstruct it
        model_data = {
            'hyperparams': self.hyperparams,
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(GB_MODEL_FILE), exist_ok=True)

            # Save model data to JSON
            with open(GB_MODEL_FILE, 'w') as f:
                json.dump(model_data, f, indent=2)

            # Save the actual model using joblib
            joblib.dump(self.model, GB_MODEL_PKL)

            print(f"Gradient Boosting model saved to {GB_MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Error saving Gradient Boosting model: {str(e)}")
            return False

    def load_model(self):
        """
        Load the model from file
        """
        if not os.path.exists(GB_MODEL_FILE) or not os.path.exists(GB_MODEL_PKL):
            print(f"Gradient Boosting model files not found")
            return False

        try:
            # Load model data from JSON
            with open(GB_MODEL_FILE, 'r') as f:
                model_data = json.load(f)

            # Set feature names
            self.feature_names = model_data['feature_names']

            # Set hyperparameters
            self.hyperparams = model_data['hyperparams']

            # Set scaler parameters if available
            if model_data['scaler_mean'] and model_data['scaler_scale']:
                self.scaler.mean_ = np.array(model_data['scaler_mean'])
                self.scaler.scale_ = np.array(model_data['scaler_scale'])
                self.scaler.n_features_in_ = len(model_data['scaler_mean'])

            # Set training metrics
            self.training_metrics = model_data['training_metrics']

            # Load the actual model using joblib
            self.model = joblib.load(GB_MODEL_PKL)

            self.is_trained = True
            print(f"Gradient Boosting model loaded from {GB_MODEL_FILE}")
            return True
        except Exception as e:
            print(f"Error loading Gradient Boosting model: {str(e)}")
            return False

    def optimize_hyperparameters(self, X, y, cv=3):
        """
        Optimize hyperparameters using grid search

        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        cv : int
            Number of cross-validation folds

        Returns:
        --------
        best_params : dict
            Best hyperparameters
        """
        print("Starting hyperparameter optimization for Gradient Boosting model...")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        # Create base model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            verbosity=0,
            random_state=42
        )

        # Create grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            verbose=1,
            n_jobs=-1
        )

        # Fit grid search
        grid_search.fit(X_scaled, y)

        # Get best parameters
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")

        # Update model with best parameters
        self.hyperparams.update(best_params)
        self.model = xgb.XGBClassifier(**self.hyperparams)

        # Train model with best parameters
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Evaluate model
        self._evaluate_model(X_scaled, y)

        # Save model
        self.save_model()

        return best_params

# Create a singleton instance for global use
gb_model = GradientBoostingModel()

def train_gradient_boosting_model(patient_data_list):
    """
    Train the Gradient Boosting model with patient data

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
    print(f"Gradient Boosting Model: Training with {num_records} patient records")

    if not patient_data_list or len(patient_data_list) < 10:
        print("Insufficient data for Gradient Boosting model training (need at least 10 records)")
        return {
            'success': False,
            'message': f"Insufficient data for Gradient Boosting model training (need at least 10 records, got {num_records})",
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

        # Extract features
        try:
            # Import the engineer_clinical_features function
            from clinical_ml_model import engineer_clinical_features

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

    if len(X) < 10:
        print(f"Insufficient labeled data for Gradient Boosting model training: only {len(X)} usable records out of {len(patient_data_list)} total")
        return {
            'success': False,
            'message': f"Insufficient labeled data for Gradient Boosting model training: only {len(X)} usable records out of {len(patient_data_list)} total",
            'num_records': len(X),
            'total_records': len(patient_data_list),
            'processed_count': processed_count,
            'skipped_count': skipped_count
        }

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Create and train the model
    global gb_model
    gb_model = GradientBoostingModel()

    # Check if we have enough data for hyperparameter optimization
    if len(X) >= 30:
        print("Performing hyperparameter optimization...")
        gb_model.optimize_hyperparameters(X, y, cv=min(5, len(X) // 6))
    else:
        print("Not enough data for hyperparameter optimization, using default parameters")
        gb_model.fit(X, y, feature_names=feature_names)

    # Record training event
    training_event = {
        'timestamp': datetime.now().isoformat(),
        'num_records': len(X),
        'total_records': len(patient_data_list),
        'processed_count': processed_count,
        'skipped_count': skipped_count,
        'metrics': gb_model.training_metrics,
        'feature_importance': gb_model.get_feature_importance(),
        'message': f"Gradient Boosting model trained successfully with {len(X)} usable records out of {len(patient_data_list)} total"
    }

    # Save training history
    save_training_history(training_event)

    return {
        'success': True,
        'message': training_event['message'],
        'metrics': gb_model.training_metrics,
        'feature_importance': gb_model.get_feature_importance()
    }

def save_training_history(training_event):
    """
    Save training history to file
    """
    # Load existing history
    if os.path.exists(GB_TRAINING_HISTORY_FILE):
        try:
            with open(GB_TRAINING_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []

    # Add new event
    history.append(training_event)

    # Save history
    os.makedirs(os.path.dirname(GB_TRAINING_HISTORY_FILE), exist_ok=True)
    with open(GB_TRAINING_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
