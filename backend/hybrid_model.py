"""
Hybrid Heart Failure Prediction Model
This module integrates the rule-based model with the ML model to create a hybrid prediction system.
"""
import os
import json
import subprocess
from datetime import datetime

# Import the component models
import model_enhancer
import clinical_ml_model

# Try to import the ML model extensions
try:
    import ml_model_extensions
    RF_MODEL_AVAILABLE = True
    print("Random Forest model extensions loaded successfully")
except ImportError:
    RF_MODEL_AVAILABLE = False
    print("Random Forest model extensions not available")

# Define paths for model data
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
HYBRID_CONFIG_FILE = os.path.join(MODEL_DIR, 'hybrid_config.json')
HYBRID_TRAINING_HISTORY_FILE = os.path.join(MODEL_DIR, 'hybrid_training_history.json')

# Default ensemble weights
DEFAULT_WEIGHTS = {
    'rule_based': 0.7,  # Higher weight for rule-based model initially
    'ml_model': 0.2,    # Lower weight for ML model initially
    'random_forest': 0.1  # Lowest weight for Random Forest initially (will be 0 if not available)
}

class HybridHeartFailurePredictor:
    def __init__(self):
        """
        Initialize the hybrid prediction model
        """
        self.ensemble_weights = self._load_weights()
        self.ml_model_available = self._check_ml_model()
        self.rf_model_available = self._check_rf_model()

        # If Random Forest is not available, set its weight to 0 and redistribute
        if not self.rf_model_available and 'random_forest' in self.ensemble_weights:
            rf_weight = self.ensemble_weights.pop('random_forest', 0)
            # Redistribute the Random Forest weight proportionally to other models
            if rf_weight > 0:
                total_remaining = sum(self.ensemble_weights.values())
                if total_remaining > 0:
                    for model in self.ensemble_weights:
                        self.ensemble_weights[model] += (rf_weight * self.ensemble_weights[model] / total_remaining)
                else:
                    # If no weights remain, give all to rule-based
                    self.ensemble_weights['rule_based'] = 1.0

    def _load_weights(self):
        """
        Load ensemble weights from file or use defaults
        """
        if os.path.exists(HYBRID_CONFIG_FILE):
            try:
                with open(HYBRID_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                return config.get('ensemble_weights', DEFAULT_WEIGHTS)
            except Exception as e:
                print(f"Error loading hybrid config: {str(e)}")

        return DEFAULT_WEIGHTS

    def _save_weights(self):
        """
        Save ensemble weights to file
        """
        config = {
            'ensemble_weights': self.ensemble_weights,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(HYBRID_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving hybrid config: {str(e)}")
            return False

    def _check_ml_model(self):
        """
        Check if the ML model is available
        """
        # Create a temporary model and try to load it
        model = clinical_ml_model.ClinicallyInformedLogisticRegression()
        return model.load_model()

    def _check_rf_model(self):
        """
        Check if the Random Forest model is available
        """
        # First check if the module is available
        if not RF_MODEL_AVAILABLE:
            return False

        # Try to load the model
        try:
            # Create a temporary model instance
            rf_model = ml_model_extensions.RandomForestHeartFailureModel()
            # Try to load the model
            return rf_model.load_model()
        except Exception as e:
            print(f"Error checking Random Forest model: {str(e)}")
            return False

    def predict(self, patient_data):
        """
        Make a prediction using the hybrid model
        """
        # Get rule-based prediction
        rule_prediction, rule_confidence, rule_shap = model_enhancer.predict_heart_failure(patient_data)

        # Initialize with rule-based prediction
        final_prediction = rule_prediction
        final_confidence = rule_confidence
        explanations = {
            'rule_based': {
                'prediction': rule_prediction,
                'confidence': rule_confidence,
                'shap_values': rule_shap
            }
        }

        # Initialize model predictions and confidences
        ml_prediction = None
        ml_confidence = None
        rf_prediction = None
        rf_confidence = None

        # If ML model is available, incorporate its prediction
        if self.ml_model_available:
            try:
                # Get ML prediction
                ml_prediction, ml_confidence, ml_explanation = clinical_ml_model.predict_heart_failure(patient_data)

                # Store ML results
                explanations['ml_model'] = {
                    'prediction': ml_prediction,
                    'confidence': ml_confidence,
                    'explanation': ml_explanation
                }
            except Exception as e:
                print(f"Error incorporating ML prediction: {str(e)}")
                ml_prediction = None
                ml_confidence = None
        else:
            print("ML model not available")

        # If Random Forest model is available, incorporate its prediction
        if self.rf_model_available:
            try:
                # Get Random Forest prediction
                rf_prediction, rf_confidence, rf_explanation = ml_model_extensions.predict_heart_failure(patient_data)

                # Store Random Forest results
                explanations['random_forest'] = {
                    'prediction': rf_prediction,
                    'confidence': rf_confidence,
                    'explanation': rf_explanation
                }
            except Exception as e:
                print(f"Error incorporating Random Forest prediction: {str(e)}")
                rf_prediction = None
                rf_confidence = None
        else:
            print("Random Forest model not available")

        # Combine predictions using ensemble weights
        weighted_sum = self.ensemble_weights['rule_based'] * rule_prediction
        weight_total = self.ensemble_weights['rule_based']

        # Add ML prediction if available
        if ml_prediction is not None:
            weighted_sum += self.ensemble_weights['ml_model'] * ml_prediction
            weight_total += self.ensemble_weights['ml_model']

        # Add Random Forest prediction if available
        if rf_prediction is not None and 'random_forest' in self.ensemble_weights:
            weighted_sum += self.ensemble_weights['random_forest'] * rf_prediction
            weight_total += self.ensemble_weights['random_forest']

        # Normalize the weighted sum
        if weight_total > 0:
            final_prediction = weighted_sum / weight_total
        else:
            final_prediction = rule_prediction

        # Calculate model agreement
        agreements = []
        confidences = [rule_confidence]

        # Add ML agreement if available
        if ml_prediction is not None:
            agreements.append(1.0 - abs(rule_prediction - ml_prediction))
            confidences.append(ml_confidence)

        # Add Random Forest agreement if available
        if rf_prediction is not None:
            agreements.append(1.0 - abs(rule_prediction - rf_prediction))
            confidences.append(rf_confidence)

            # Also add agreement between ML and RF if both available
            if ml_prediction is not None:
                agreements.append(1.0 - abs(ml_prediction - rf_prediction))

        # Calculate average agreement and confidence
        avg_agreement = sum(agreements) / len(agreements) if agreements else 1.0
        avg_confidence = sum(confidences) / len(confidences)

        # Final confidence is a combination of average confidence and agreement
        final_confidence = (avg_confidence + avg_agreement) / 2

        # Add ensemble information to explanations
        explanations['ensemble'] = {
            'weights': self.ensemble_weights,
            'agreement': avg_agreement,
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'models_used': {
                'rule_based': True,
                'ml_model': ml_prediction is not None,
                'random_forest': rf_prediction is not None
            }
        }

        # Log the prediction
        log_msg = f"Hybrid prediction: {final_prediction:.4f} (Rule: {rule_prediction:.4f}"
        if ml_prediction is not None:
            log_msg += f", ML: {ml_prediction:.4f}"
        if rf_prediction is not None:
            log_msg += f", RF: {rf_prediction:.4f}"
        log_msg += f") [Agreement: {avg_agreement:.2f}]"
        print(log_msg)

        return final_prediction, final_confidence, explanations

    def retrain(self, patient_data_list=None):
        """
        Retrain both models and update ensemble weights
        """
        results = {
            'success': False,
            'rule_based_result': None,
            'ml_model_result': None,
            'random_forest_result': None,
            'ensemble_weights': self.ensemble_weights,
            'message': ""
        }

        # If no patient data provided, try to load it
        if not patient_data_list:
            print("No patient data provided, loading from files...")
            patient_data_list = self._load_patient_data()

        # Count the number of records
        num_records = len(patient_data_list) if patient_data_list else 0
        print(f"Retraining with {num_records} patient records")

        # Store the count in the results
        results['num_records'] = num_records
        results['total_records'] = num_records  # Track total records before filtering

        if not patient_data_list:
            results['message'] = "No patient data available for retraining"
            return results

        # Retrain rule-based model
        print("Retraining rule-based model...")
        rule_result = model_enhancer.retrain_model()
        results['rule_based_result'] = rule_result

        # Retrain ML model
        print("Retraining ML model...")
        ml_result = clinical_ml_model.train_ml_model(patient_data_list)
        results['ml_model_result'] = ml_result

        # Update results with detailed information from ML model
        if 'total_records' in ml_result:
            results['total_records'] = ml_result['total_records']
        if 'processed_count' in ml_result:
            results['processed_count'] = ml_result['processed_count']
        if 'skipped_count' in ml_result:
            results['skipped_count'] = ml_result['skipped_count']

        # Retrain Random Forest model if available
        rf_result = {'success': False, 'message': 'Random Forest model not available'}
        if RF_MODEL_AVAILABLE:
            try:
                print("Retraining Random Forest model...")
                rf_result = ml_model_extensions.train_random_forest_model(patient_data_list)
                print(f"Random Forest training result: {rf_result['message']}")
            except Exception as e:
                print(f"Error training Random Forest model: {str(e)}")
                rf_result = {'success': False, 'message': f"Error training Random Forest model: {str(e)}"}

        results['random_forest_result'] = rf_result

        # Update ensemble weights based on training results
        self._update_ensemble_weights(rule_result, ml_result, rf_result, len(patient_data_list))
        results['ensemble_weights'] = self.ensemble_weights

        # Set success status based on results
        if rule_result['success'] and (ml_result['success'] or rf_result['success']):
            results['success'] = True
            models_trained = []
            if rule_result['success']:
                models_trained.append("rule-based")
            if ml_result['success']:
                models_trained.append("logistic regression")
            if rf_result['success']:
                models_trained.append("random forest")

            results['message'] = f"Models retrained successfully ({', '.join(models_trained)}) with {ml_result.get('num_records', 0)} usable records"
        elif rule_result['success']:
            results['success'] = True
            results['message'] = f"Only rule-based model retrained successfully. ML models failed."
        else:
            results['success'] = False
            results['message'] = f"Failed to retrain models."

        # Save updated weights
        self._save_weights()

        # Check if models are now available
        self.ml_model_available = self._check_ml_model()
        self.rf_model_available = self._check_rf_model()

        # Record training event
        training_event = {
            'timestamp': datetime.now().isoformat(),
            'num_records': ml_result.get('num_records', len(patient_data_list)),
            'total_records': ml_result.get('total_records', len(patient_data_list)),
            'processed_count': ml_result.get('processed_count', 0),
            'skipped_count': ml_result.get('skipped_count', 0),
            'rule_based_result': rule_result,
            'ml_model_result': ml_result,
            'random_forest_result': rf_result,
            'ensemble_weights': self.ensemble_weights,
            'models_available': {
                'rule_based': True,
                'ml_model': self.ml_model_available,
                'random_forest': self.rf_model_available
            },
            'message': results['message']
        }

        self._save_training_event(training_event)

        # Set success flag and message
        results['success'] = rule_result.get('success', False) or ml_result.get('success', False)
        results['message'] = f"Hybrid model retrained with {len(patient_data_list)} records"
        results['num_records'] = len(patient_data_list)

        return results

    def _update_ensemble_weights(self, rule_result, ml_result, rf_result, num_records):
        """
        Update ensemble weights based on training results
        """
        # Start with default weights
        weights = {
            'rule_based': 0.6,
            'ml_model': 0.2,
            'random_forest': 0.2
        }

        # Track which models are available and successful
        ml_available = ml_result.get('success', False)
        rf_available = rf_result.get('success', False)

        # If Random Forest is not available, remove it from weights
        if not rf_available:
            weights.pop('random_forest', None)

        # Adjust metrics based on number of records (more records = more trust in ML models)
        data_factor = min(0.6, num_records / 100)  # Cap at 0.6
        # We'll use this factor to adjust metrics later

        # Calculate performance metrics for each model
        model_metrics = {}

        # ML model metrics
        if ml_available:
            metrics = ml_result.get('metrics', {})
            model_metrics['ml_model'] = metrics.get('roc_auc', 0.5)

            # Bonus for good performance
            if model_metrics['ml_model'] > 0.7:
                model_metrics['ml_model'] += 0.1
        else:
            model_metrics['ml_model'] = 0

        # Random Forest metrics
        if rf_available:
            metrics = rf_result.get('metrics', {})
            model_metrics['random_forest'] = metrics.get('roc_auc', 0.5)

            # Bonus for good performance
            if model_metrics['random_forest'] > 0.7:
                model_metrics['random_forest'] += 0.1
        else:
            model_metrics['random_forest'] = 0

        # Rule-based model gets a baseline score that decreases as we get more data
        # This gives ML models more weight as we collect more training data
        model_metrics['rule_based'] = max(0.4, 0.7 - data_factor)

        # Calculate total metric score
        total_metric_score = sum(model_metrics.values())

        # Distribute weights based on metrics if we have valid metrics
        if total_metric_score > 0:
            # Calculate weights proportional to metrics
            for model in model_metrics:
                if model in weights:  # Only set weights for models that are in our weights dict
                    weights[model] = model_metrics[model] / total_metric_score
        else:
            # Default weights if no metrics available
            if rf_available and ml_available:
                weights = {'rule_based': 0.6, 'ml_model': 0.2, 'random_forest': 0.2}
            elif ml_available:
                weights = {'rule_based': 0.7, 'ml_model': 0.3}
            elif rf_available:
                weights = {'rule_based': 0.7, 'random_forest': 0.3}
            else:
                weights = {'rule_based': 1.0}

        # Ensure weights sum to 1.0
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            for model in weights:
                weights[model] /= weight_sum

        # Log the updated weights
        weight_str = ", ".join([f"{model}={weights[model]:.2f}" for model in weights])
        print(f"Updated ensemble weights: {weight_str}")

        self.ensemble_weights = weights

    def _load_patient_data(self):
        """
        Load patient data from files
        """
        patient_data_list = []
        if os.path.exists('data/patients'):
            # Get absolute path to avoid caching issues
            patient_dir = os.path.abspath('data/patients')
            print(f"Loading patient data from directory: {patient_dir}")

            # Force a file system sync to ensure all changes are visible
            try:
                if hasattr(os, 'sync'):
                    os.sync()
                # Alternative approach for systems without os.sync
                subprocess.run(['sync'], check=False)
                print("File system synced to ensure latest changes are visible")
            except Exception as sync_error:
                print(f"Warning: Could not sync file system: {str(sync_error)}")

            # Get all patient files
            try:
                patient_files = [f for f in os.listdir(patient_dir) if f.endswith('.json')]
                print(f"Found {len(patient_files)} patient files for training")

                # Debug: print all files
                for i, file in enumerate(patient_files):
                    file_path = os.path.join(patient_dir, file)
                    file_size = os.path.getsize(file_path)
                    file_mtime = os.path.getmtime(file_path)
                    print(f"  {i+1}. {file} (size: {file_size} bytes, modified: {datetime.fromtimestamp(file_mtime).isoformat()})")
            except Exception as e:
                print(f"Error listing patient files: {str(e)}")
                import traceback
                traceback.print_exc()
                patient_files = []

            # Sort files by modification time (newest first) to ensure consistent ordering
            sorted_files = sorted(
                [(f, os.path.getmtime(os.path.join(patient_dir, f))) for f in patient_files],
                key=lambda x: x[1],
                reverse=True
            )

            # Process each file
            for filename, mtime in sorted_files:
                try:
                    file_path = os.path.join(patient_dir, filename)
                    print(f"Loading patient data from {file_path} (modified: {datetime.fromtimestamp(mtime).isoformat()})")

                    with open(file_path, 'r') as f:
                        patient = json.load(f)

                    # Verify the patient data has the required structure
                    if 'patient_data' not in patient:
                        print(f"Warning: File {filename} is missing 'patient_data' field, skipping")
                        continue

                    patient_data_list.append(patient)
                    print(f"Successfully loaded patient ID: {patient.get('patient_id', 'unknown')}")
                except Exception as e:
                    print(f"Error loading patient data from {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        print(f"Total patient records loaded for training: {len(patient_data_list)}")
        return patient_data_list

    def _save_training_event(self, training_event):
        """
        Save a training event to the training history
        """
        # Load existing history
        history = []
        if os.path.exists(HYBRID_TRAINING_HISTORY_FILE):
            try:
                with open(HYBRID_TRAINING_HISTORY_FILE, 'r') as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            except Exception as e:
                print(f"Error loading hybrid training history: {str(e)}")

        # Add new event
        history.append(training_event)

        # Save updated history
        try:
            with open(HYBRID_TRAINING_HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"Saved hybrid training event with {training_event['num_records']} records")
            return True
        except Exception as e:
            print(f"Error saving hybrid training event: {str(e)}")
            return False

    def get_training_history(self):
        """
        Get the hybrid model training history
        """
        # First check for the new training history file
        if os.path.exists('data/training_history.json'):
            print("Loading training history from data/training_history.json")
            try:
                with open('data/training_history.json', 'r') as f:
                    history = json.load(f)

                # Ensure it's a list
                if not isinstance(history, list):
                    history = [history]

                # Sort by timestamp (newest first)
                history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

                print(f"Returning {len(history)} training history entries")
                return history
            except Exception as e:
                print(f"Error loading training history from data/training_history.json: {str(e)}")

        # Fall back to the hybrid training history file
        if os.path.exists(HYBRID_TRAINING_HISTORY_FILE):
            try:
                with open(HYBRID_TRAINING_HISTORY_FILE, 'r') as f:
                    history = json.load(f)

                # Ensure it's a list
                if not isinstance(history, list):
                    history = [history]

                print(f"Returning {len(history)} hybrid training history entries")
                return history
            except Exception as e:
                print(f"Error loading hybrid training history: {str(e)}")

        print("No training history found")
        return []

# Create a singleton instance
hybrid_model = HybridHeartFailurePredictor()

def predict_heart_failure(patient_data):
    """
    Predict heart failure using the hybrid model
    """
    return hybrid_model.predict(patient_data)

def retrain_model(patient_data_list=None):
    """
    Retrain the hybrid model
    """
    return hybrid_model.retrain(patient_data_list)

def get_training_history():
    """
    Get the hybrid model training history
    """
    return hybrid_model.get_training_history()
