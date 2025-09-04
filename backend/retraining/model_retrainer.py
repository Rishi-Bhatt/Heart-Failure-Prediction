import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config manager
try:
    from config.config_manager import get_training_config
except ImportError:
    # Fallback function if config_manager is not available
    def get_training_config():
        return {
            'epochs': 50,
            'use_neural_network': False,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }

# Try to import neural network model, but make it optional
NeuralNetworkModel = None
try:
    from models.neural_network_model import NeuralNetworkModel
except ImportError:
    print("Neural network model not available. Will use Random Forest only.")

class ModelRetrainer:
    """
    Class for handling model retraining
    """

    def __init__(self, model, retraining_threshold=20):
        """
        Initialize the model retrainer

        Parameters:
        -----------
        model : HeartFailureModel
            The heart failure prediction model
        retraining_threshold : int
            Number of new records before retraining
        """
        self.model = model
        self.retraining_threshold = retraining_threshold
        self.retraining_history_path = 'data/retraining_history.json'
        self.drift_detection_threshold = 0.1  # Performance drop threshold for drift detection

        # Load training configuration
        self._load_training_config()

        # Initialize neural network model if enabled and available
        if self.use_neural_network and NeuralNetworkModel is not None:
            try:
                self.nn_model = NeuralNetworkModel(input_dim=len(self.model.feature_names))
                print("Neural network model initialized")
            except Exception as e:
                print(f"Error initializing neural network model: {str(e)}")
                self.use_neural_network = False
        else:
            # Neural network not available or not enabled
            self.use_neural_network = False

    def _load_training_config(self):
        """
        Load training configuration from config file
        """
        try:
            config = get_training_config()
            self.epochs = config.get('epochs', 50)
            self.use_neural_network = config.get('use_neural_network', False)
            self.batch_size = config.get('batch_size', 32)
            self.validation_split = config.get('validation_split', 0.2)
            self.early_stopping_patience = config.get('early_stopping_patience', 10)
            self.learning_rate = config.get('learning_rate', 0.001)
            self.optimizer = config.get('optimizer', 'adam')
            print(f"Loaded training config: epochs={self.epochs}, use_neural_network={self.use_neural_network}")
        except Exception as e:
            print(f"Error loading training config: {str(e)}")
            # Use default values
            self.epochs = 50
            self.use_neural_network = False
            self.batch_size = 32
            self.validation_split = 0.2
            self.early_stopping_patience = 10
            self.learning_rate = 0.001
            self.optimizer = 'adam'

        # Initialize or load retraining history
        try:
            if os.path.exists(self.retraining_history_path):
                with open(self.retraining_history_path, 'r') as f:
                    self.retraining_history = json.load(f)

                # Ensure all required fields exist
                if 'last_retraining_date' not in self.retraining_history:
                    self.retraining_history['last_retraining_date'] = datetime.now().isoformat()
                if 'retraining_count' not in self.retraining_history:
                    self.retraining_history['retraining_count'] = 0
                if 'records_since_last_retraining' not in self.retraining_history:
                    self.retraining_history['records_since_last_retraining'] = 0
                if 'performance_history' not in self.retraining_history:
                    self.retraining_history['performance_history'] = []
                if 'drift_detected' not in self.retraining_history:
                    self.retraining_history['drift_detected'] = False
            else:
                self.retraining_history = {
                    'last_retraining_date': datetime.now().isoformat(),
                    'retraining_count': 0,
                    'records_since_last_retraining': 0,
                    'performance_history': [],
                    'drift_detected': False
                }
            self._save_retraining_history()
        except Exception as e:
            print(f"Error initializing retraining history: {str(e)}")
            # Create a new retraining history if there's an error
            self.retraining_history = {
                'last_retraining_date': datetime.now().isoformat(),
                'retraining_count': 0,
                'records_since_last_retraining': 0,
                'performance_history': [],
                'drift_detected': False
            }
            self._save_retraining_history()

    def check_retraining(self):
        """
        Check if model retraining is needed

        Returns:
        --------
        retraining_info : dict
            Information about retraining status
        """
        # Increment records count
        self.retraining_history['records_since_last_retraining'] += 1
        self._save_retraining_history()

        # Check for drift
        self._detect_drift()

        # Determine if retraining is needed
        retraining_needed = (
            self.retraining_history['records_since_last_retraining'] >= self.retraining_threshold or
            self.retraining_history['drift_detected']
        )

        retraining_info = {
            'retraining_needed': retraining_needed,
            'records_since_last_retraining': self.retraining_history['records_since_last_retraining'],
            'retraining_threshold': self.retraining_threshold,
            'drift_detected': self.retraining_history['drift_detected']
        }

        # If retraining is needed, retrain the model
        if retraining_needed:
            self.retrain()

        return retraining_info

    def retrain(self):
        """
        Retrain the model using all available data

        Returns:
        --------
        retraining_result : dict
            Results of the retraining process
        """
        print("Retraining model...")

        # Collect all patient data
        patient_data = self._collect_patient_data()

        if not patient_data or len(patient_data) < 5:
            return {
                'success': False,
                'message': 'Not enough data for retraining',
                'date': datetime.now().isoformat()
            }

        # Prepare data for training
        X, y = self._prepare_training_data(patient_data)

        if len(X) == 0 or len(y) == 0:
            return {
                'success': False,
                'message': 'Failed to prepare training data',
                'date': datetime.now().isoformat()
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.model.scaler.fit_transform(X_train)
        X_test_scaled = self.model.scaler.transform(X_test)

        # Train models
        metrics = {}
        training_plots = {}

        # Train Random Forest model
        self.model.model = RandomForestClassifier(
            n_estimators=200,           # Increased from 100
            max_depth=15,               # Increased from 10
            min_samples_split=5,        # More conservative split criterion
            min_samples_leaf=2,         # Slightly more conservative
            max_features='sqrt',        # Use sqrt of features for each tree
            bootstrap=True,             # Use bootstrapping
            class_weight='balanced',    # Handle class imbalance
            random_state=42
        )
        self.model.model.fit(X_train_scaled, y_train)

        # Evaluate Random Forest model
        y_pred_rf = self.model.model.predict(X_test_scaled)
        y_prob_rf = self.model.model.predict_proba(X_test_scaled)[:, 1]

        metrics['random_forest'] = {
            'accuracy': float(accuracy_score(y_test, y_pred_rf)),
            'precision': float(precision_score(y_test, y_pred_rf, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_rf, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_rf, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_prob_rf))
        }

        # Train Neural Network model if enabled, available, and we have enough data
        if self.use_neural_network and NeuralNetworkModel is not None and len(X_train) >= 20:
            try:
                # Reload training config to get latest settings
                self._load_training_config()

                # Check if neural network model is initialized
                if not hasattr(self, 'nn_model') or self.nn_model is None:
                    print("Neural network model not initialized. Skipping neural network training.")
                else:
                    print(f"Training neural network with {self.epochs} epochs...")
                    # Convert to numpy arrays
                    X_train_np = X_train_scaled.values if hasattr(X_train_scaled, 'values') else X_train_scaled
                    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
                    X_test_np = X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled
                    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test

                    # Train neural network with configuration parameters
                    history = self.nn_model.fit(
                        X_train_np, y_train_np,
                        epochs=self.epochs,
                        batch_size=min(self.batch_size, len(X_train_np)),  # Adjust batch size for small datasets
                        validation_split=self.validation_split,
                        verbose=1
                    )

                    # Evaluate neural network
                    nn_metrics = self.nn_model.evaluate(X_test_np, y_test_np)
                    metrics['neural_network'] = nn_metrics

                    # Generate training plot
                    training_plots['neural_network'] = self.nn_model.get_training_plot(history)

                    print(f"Neural network trained with accuracy: {nn_metrics['accuracy']:.4f}")
            except Exception as e:
                print(f"Error training neural network: {str(e)}")

        # Log the best model's metrics
        if 'neural_network' in metrics and metrics['neural_network']['accuracy'] > metrics['random_forest']['accuracy']:
            print(f"Neural network model performed better: {metrics['neural_network']['accuracy']:.4f} vs {metrics['random_forest']['accuracy']:.4f}")
        else:
            print(f"Random forest model performed better: {metrics['random_forest']['accuracy']:.4f}")

        # Save model
        joblib.dump(self.model.model, self.model.model_path)
        joblib.dump(self.model.scaler, self.model.scaler_path)

        # Update retraining history
        self.retraining_history['last_retraining_date'] = datetime.now().isoformat()
        self.retraining_history['retraining_count'] += 1
        self.retraining_history['records_since_last_retraining'] = 0
        self.retraining_history['drift_detected'] = False

        # Add number of records used for training
        num_records = len(patient_data)

        # Add entry to performance history
        self.retraining_history['performance_history'].append({
            'date': datetime.now().isoformat(),
            'metrics': metrics,
            'num_records': num_records,
            'epochs': self.epochs if self.use_neural_network else None
        })
        self._save_retraining_history()

        # Create retraining result
        retraining_result = {
            'success': True,
            'message': f'Model retrained successfully with {num_records} records',
            'date': datetime.now().isoformat(),
            'metrics': metrics,
            'num_records': num_records,
            'epochs': self.epochs if self.use_neural_network else None
        }

        # Add training plots if available
        if training_plots:
            retraining_result['training_plots'] = training_plots

        # Print summary of results
        if 'neural_network' in metrics:
            print(f"Neural network accuracy: {metrics['neural_network']['accuracy']:.4f}")
        print(f"Random forest accuracy: {metrics['random_forest']['accuracy']:.4f}")

        return retraining_result

    def _collect_patient_data(self):
        """
        Collect all patient data for retraining

        Returns:
        --------
        patient_data : list
            List of patient data dictionaries
        """
        patient_data = []

        # Check if patients directory exists
        if not os.path.exists('data/patients'):
            return patient_data

        # Load all patient data
        for filename in os.listdir('data/patients'):
            if filename.endswith('.json'):
                try:
                    with open(f'data/patients/{filename}', 'r') as f:
                        data = json.load(f)
                        patient_data.append(data)
                except Exception as e:
                    print(f"Error loading patient data from {filename}: {str(e)}")

        return patient_data

    def _prepare_training_data(self, patient_data):
        """
        Prepare training data from patient records

        Parameters:
        -----------
        patient_data : list
            List of patient data dictionaries

        Returns:
        --------
        X : pandas.DataFrame
            Feature matrix
        y : pandas.Series
            Target vector
        """
        # Create empty lists for features and target
        features_list = []
        target_list = []

        # Process each patient record
        for patient in patient_data:
            try:
                # Extract features
                features = {}

                # Check if patient_data exists
                if 'patient_data' not in patient:
                    continue

                # Basic clinical features from patient_data
                patient_info = patient['patient_data']

                # Set default values for all features
                features['age'] = 60
                features['sex'] = 0
                features['chest_pain_type'] = 0
                features['resting_bp'] = 120
                features['cholesterol'] = 200
                features['fasting_blood_sugar'] = 0
                features['resting_ecg'] = 0
                features['max_heart_rate'] = 150
                features['exercise_induced_angina'] = 0
                features['st_depression'] = 0.0
                features['st_slope'] = 0
                features['num_major_vessels'] = 0
                features['thalassemia'] = 0
                features['prior_event_severity'] = 0
                features['time_since_event'] = 24
                features['pvc_count'] = 0
                features['qt_prolongation'] = 0
                features['af_detected'] = 0
                features['tachycardia_detected'] = 0
                features['bradycardia_detected'] = 0

                # Advanced features (derived from existing data)
                features['age_squared'] = 3600  # Default age^2
                features['bp_age_ratio'] = 2.0   # Default resting_bp/age
                features['heart_rate_recovery'] = 0  # Default
                features['cholesterol_hdl_ratio'] = 4.0  # Default
                features['bmi'] = 25.0  # Default

                # Update with actual values if available
                features['age'] = patient_info.get('age', 60)
                features['sex'] = 1 if patient_info.get('gender', 'Male') == 'Male' else 0

                try:
                    features['chest_pain_type'] = self.model._encode_chest_pain(patient_info.get('chest_pain_type', 'Typical Angina'))
                except:
                    features['chest_pain_type'] = 0

                # Blood pressure - extract systolic
                try:
                    bp = patient_info.get('blood_pressure', '120/80')
                    if isinstance(bp, str) and '/' in bp:
                        systolic = int(bp.split('/')[0])
                        features['resting_bp'] = systolic
                    elif isinstance(bp, (int, float)):
                        features['resting_bp'] = bp
                    else:
                        features['resting_bp'] = 120
                except:
                    features['resting_bp'] = 120

                # Handle numeric features
                try:
                    chol = patient_info.get('cholesterol', 200)
                    if isinstance(chol, str):
                        # Handle empty strings
                        if chol.strip() == '':
                            features['cholesterol'] = 200
                        else:
                            features['cholesterol'] = int(float(chol))
                    elif isinstance(chol, (int, float)):
                        features['cholesterol'] = int(chol)
                    else:
                        features['cholesterol'] = 200
                except:
                    features['cholesterol'] = 200

                try:
                    fbs = patient_info.get('fasting_blood_sugar', 120)
                    if isinstance(fbs, str):
                        # Handle empty strings
                        if fbs.strip() == '':
                            features['fasting_blood_sugar'] = 0
                        else:
                            fbs = float(fbs)
                            features['fasting_blood_sugar'] = 1 if fbs > 120 else 0
                    elif isinstance(fbs, (int, float)):
                        features['fasting_blood_sugar'] = 1 if fbs > 120 else 0
                    else:
                        features['fasting_blood_sugar'] = 0
                except:
                    features['fasting_blood_sugar'] = 0

                try:
                    features['resting_ecg'] = self.model._encode_resting_ecg(patient_info.get('ecg_result', 'Normal'))
                except:
                    features['resting_ecg'] = 0

                try:
                    hr = patient_info.get('max_heart_rate', 150)
                    if isinstance(hr, str):
                        # Handle empty strings
                        if hr.strip() == '':
                            features['max_heart_rate'] = 150
                        else:
                            features['max_heart_rate'] = int(float(hr))
                    elif isinstance(hr, (int, float)):
                        features['max_heart_rate'] = int(hr)
                    else:
                        features['max_heart_rate'] = 150
                except:
                    features['max_heart_rate'] = 150

                features['exercise_induced_angina'] = 1 if patient_info.get('exercise_induced_angina', False) else 0

                try:
                    st_dep = patient_info.get('st_depression', 0.0)
                    if isinstance(st_dep, str):
                        # Handle empty strings
                        if st_dep.strip() == '':
                            features['st_depression'] = 0.0
                        else:
                            features['st_depression'] = float(st_dep)
                    elif isinstance(st_dep, (int, float)):
                        features['st_depression'] = float(st_dep)
                    else:
                        features['st_depression'] = 0.0
                except:
                    features['st_depression'] = 0.0

                try:
                    features['st_slope'] = self.model._encode_st_slope(patient_info.get('slope_of_st', 'Flat'))
                except:
                    features['st_slope'] = 0

                try:
                    vessels = patient_info.get('number_of_major_vessels', 0)
                    if isinstance(vessels, str):
                        # Handle empty strings
                        if vessels.strip() == '':
                            features['num_major_vessels'] = 0
                        else:
                            features['num_major_vessels'] = int(float(vessels))
                    elif isinstance(vessels, (int, float)):
                        features['num_major_vessels'] = int(vessels)
                    else:
                        features['num_major_vessels'] = 0
                except:
                    features['num_major_vessels'] = 0

                try:
                    features['thalassemia'] = self.model._encode_thalassemia(patient_info.get('thalassemia', 'Normal'))
                except:
                    features['thalassemia'] = 0

                # Prior cardiac event features
                prior_event = patient_info.get('prior_cardiac_event', {})
                if prior_event:
                    severity = prior_event.get('severity', 'Mild')
                    features['prior_event_severity'] = {'Mild': 1, 'Moderate': 2, 'Severe': 3}.get(severity, 0)
                    features['time_since_event'] = prior_event.get('time_since_event', 12)  # in months
                else:
                    features['prior_event_severity'] = 0
                    features['time_since_event'] = 24  # default to 2 years if no event

                # ECG abnormality features
                if 'abnormalities' in patient:
                    abnormalities = patient['abnormalities']
                    features['pvc_count'] = len(abnormalities.get('PVCs', []))
                    features['qt_prolongation'] = 1 if abnormalities.get('QT_prolongation', []) else 0
                    features['af_detected'] = 1 if abnormalities.get('Atrial_Fibrillation', []) else 0
                    features['tachycardia_detected'] = 1 if abnormalities.get('Tachycardia', []) else 0
                    features['bradycardia_detected'] = 1 if abnormalities.get('Bradycardia', []) else 0

                # Calculate derived features
                try:
                    # Age squared (non-linear age effect)
                    features['age_squared'] = features['age'] ** 2

                    # Blood pressure to age ratio (higher is worse)
                    if features['age'] > 0:
                        features['bp_age_ratio'] = features['resting_bp'] / features['age']
                    else:
                        features['bp_age_ratio'] = 2.0

                    # Heart rate recovery estimation
                    if 'heart_rate_recovery' in patient_info:
                        features['heart_rate_recovery'] = float(patient_info['heart_rate_recovery'])
                    else:
                        # Estimate from max heart rate if available
                        features['heart_rate_recovery'] = max(0, features['max_heart_rate'] - 120) / 10

                    # Cholesterol HDL ratio
                    if 'hdl' in patient_info and patient_info['hdl'] and float(patient_info['hdl']) > 0:
                        hdl = float(patient_info['hdl'])
                        features['cholesterol_hdl_ratio'] = features['cholesterol'] / hdl
                    else:
                        features['cholesterol_hdl_ratio'] = 4.0

                    # BMI calculation
                    if 'weight' in patient_info and 'height' in patient_info:
                        try:
                            weight = float(patient_info['weight'])
                            height = float(patient_info['height'])
                            if height > 0:
                                # Convert height from cm to m if needed
                                if height > 3:  # Assuming height in cm if > 3
                                    height = height / 100
                                features['bmi'] = weight / (height * height)
                            else:
                                features['bmi'] = 25.0
                        except:
                            features['bmi'] = 25.0
                    else:
                        features['bmi'] = 25.0
                except Exception as e:
                    print(f"Error calculating derived features: {str(e)}")

                # Add to features list
                features_list.append(features)

                # Use the prediction as the target
                # This is a form of pseudo-labeling, assuming our model's predictions are reasonable
                try:
                    prediction = patient.get('prediction', 0.5)
                    if isinstance(prediction, str):
                        # Handle empty strings
                        if prediction.strip() == '':
                            prediction = 0.5
                        else:
                            prediction = float(prediction)
                    target_list.append(1 if prediction >= 0.5 else 0)
                except Exception as e:
                    # Log the error for debugging
                    print(f"Error converting prediction to float: {str(e)}")
                    # Default to negative class if prediction can't be parsed
                    target_list.append(0)

            except Exception as e:
                print(f"Error processing patient for training: {str(e)}")

        # Convert to DataFrame
        if features_list:
            X = pd.DataFrame(features_list)

            # Ensure all feature names are present
            for feature in self.model.feature_names:
                if feature not in X.columns:
                    X[feature] = 0

            # Reorder columns to match model's expected order
            X = X[self.model.feature_names]

            y = pd.Series(target_list)

            return X, y
        else:
            return pd.DataFrame(), pd.Series()

    def _detect_drift(self):
        """
        Detect model drift by evaluating performance on recent data
        """
        # Need at least two performance records to detect drift
        if len(self.retraining_history['performance_history']) < 2:
            return

        # Get most recent performance metrics
        latest_metrics = self.retraining_history['performance_history'][-1]['metrics']

        # Calculate average performance from previous records
        previous_metrics = [record['metrics'] for record in self.retraining_history['performance_history'][:-1]]
        avg_previous_accuracy = np.mean([m['accuracy'] for m in previous_metrics])

        # Check if current performance is significantly worse
        if avg_previous_accuracy - latest_metrics['accuracy'] > self.drift_detection_threshold:
            self.retraining_history['drift_detected'] = True
            print(f"Model drift detected! Previous accuracy: {avg_previous_accuracy:.4f}, Current: {latest_metrics['accuracy']:.4f}")
            self._save_retraining_history()

    def _save_retraining_history(self):
        """
        Save retraining history to file
        """
        os.makedirs('data', exist_ok=True)
        with open(self.retraining_history_path, 'w') as f:
            json.dump(self.retraining_history, f, indent=2)
