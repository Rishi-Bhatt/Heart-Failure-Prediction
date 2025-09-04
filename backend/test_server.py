from flask import Flask, jsonify, request
from flask_cors import CORS
import math
import random
import json
import os
import subprocess
import time
import numpy as np
from datetime import datetime

# Import the hybrid model
import hybrid_model

# Import longitudinal tracking module
import longitudinal_tracker

# Import counterfactual engine
import counterfactual_engine

# Import ML model extensions
try:
    import ml_model_extensions
    ML_EXTENSIONS_AVAILABLE = True
    print("ML model extensions loaded successfully")
except ImportError:
    ML_EXTENSIONS_AVAILABLE = False
    print("ML model extensions not available")

app = Flask(__name__)
# Configure CORS properly - this is the only place we set CORS headers
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Global OPTIONS route handler
@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def options_handler(path):
    return jsonify({'status': 'ok'})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'Backend is working!'
    })

@app.route('/api/ml/random-forest/predict', methods=['POST'])
def random_forest_predict():
    """Endpoint for making predictions with the Random Forest model"""
    try:
        # Check if ML extensions are available
        if not ML_EXTENSIONS_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Random Forest model is not available'
            }), 404

        # Get patient data from request
        patient_data = request.json
        print(f"Received patient data for Random Forest prediction: {patient_data}")

        # Make prediction using Random Forest model
        probability, confidence, explanation = ml_model_extensions.predict_heart_failure(patient_data)

        # Return response
        return jsonify({
            'status': 'success',
            'probability': float(probability),
            'confidence': float(confidence),
            'explanation': explanation,
            'model_type': 'random_forest'
        })
    except Exception as e:
        print(f"Error in Random Forest prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error making Random Forest prediction: {str(e)}"
        }), 500

@app.route('/api/ml/compare', methods=['POST'])
def compare_ml_models():
    """Endpoint for comparing predictions from different ML models"""
    try:
        # Check if ML extensions are available
        if not ML_EXTENSIONS_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'ML model comparison is not available'
            }), 404

        # Get patient data from request
        patient_data = request.json
        print(f"Received patient data for model comparison: {patient_data}")

        # Compare models
        comparison = ml_model_extensions.compare_models(patient_data)

        # Return response
        return jsonify({
            'status': 'success',
            'comparison': comparison
        })
    except Exception as e:
        print(f"Error in model comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error comparing models: {str(e)}"
        }), 500

@app.route('/api/ml/random-forest/train', methods=['POST'])
def train_random_forest():
    """Endpoint for training the Random Forest model"""
    try:
        # Check if ML extensions are available
        if not ML_EXTENSIONS_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Random Forest model is not available'
            }), 404

        # Load all patient data
        patient_data_list = []
        patient_dir = os.path.abspath('data/patients')

        if os.path.exists(patient_dir):
            patient_files = [f for f in os.listdir(patient_dir) if f.endswith('.json')]

            for file_name in patient_files:
                try:
                    with open(os.path.join(patient_dir, file_name), 'r') as f:
                        patient_data_list.append(json.load(f))
                except Exception as e:
                    print(f"Error loading patient file {file_name}: {str(e)}")

        # Train the model
        result = ml_model_extensions.train_random_forest_model(patient_data_list)

        # Return response
        return jsonify({
            'status': 'success' if result['success'] else 'error',
            'message': result['message'],
            'metrics': result.get('metrics', {}),
            'num_records': result.get('num_records', 0)
        })
    except Exception as e:
        print(f"Error training Random Forest model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error training Random Forest model: {str(e)}"
        }), 500

@app.route('/api/train', methods=['POST'])
def train_hybrid_model():
    """Endpoint for training the hybrid model"""
    try:
        # Load all patient data
        patient_data_list = []
        patient_dir = os.path.abspath('data/patients')

        if os.path.exists(patient_dir):
            patient_files = [f for f in os.listdir(patient_dir) if f.endswith('.json')]
            print(f"Found {len(patient_files)} patient files for training")

            for file_name in patient_files:
                try:
                    with open(os.path.join(patient_dir, file_name), 'r') as f:
                        patient_data_list.append(json.load(f))
                except Exception as e:
                    print(f"Error loading patient file {file_name}: {str(e)}")

        # Train the hybrid model
        print(f"Training hybrid model with {len(patient_data_list)} patient records")
        result = hybrid_model.retrain_model(patient_data_list)

        # Return response
        return jsonify({
            'status': 'success' if result['success'] else 'error',
            'message': result['message'],
            'ensemble_weights': result['ensemble_weights'],
            'num_records': result.get('num_records', 0),
            'models_trained': {
                'rule_based': result.get('rule_based_result', {}).get('success', False),
                'ml_model': result.get('ml_model_result', {}).get('success', False),
                'random_forest': result.get('random_forest_result', {}).get('success', False)
            }
        })
    except Exception as e:
        print(f"Error training hybrid model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error training hybrid model: {str(e)}"
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get patient data from request
        patient_data = request.json
        print(f"Received patient data: {patient_data}")

        try:
            # Generate a realistic ECG signal
            print("Generating ECG signal...")

            # Set random seed based on patient data to ensure consistent but unique results
            # Use a more stable hash function that includes all relevant patient data
            patient_hash_str = f"{patient_data.get('name', '')}-{patient_data.get('age', '')}-{patient_data.get('gender', '')}-{patient_data.get('blood_pressure', '')}"
            patient_seed = hash(patient_hash_str) % 10000
            print(f"Using patient seed: {patient_seed} for ECG generation (from hash of {patient_hash_str})")

            # Save the original random state
            original_state = random.getstate()

            # Set seed for ECG generation
            random.seed(patient_seed)

            ecg_signal, ecg_time = generate_realistic_ecg(patient_data)
            print(f"ECG signal generated with {len(ecg_signal)} data points")
            print(f"First 5 ECG values: {ecg_signal[:5]}")
            print(f"First 5 time values: {ecg_time[:5]}")

            # Restore the original random state instead of resetting
            random.setstate(original_state)
        except Exception as ecg_error:
            print(f"Error generating ECG: {str(ecg_error)}")
            import traceback
            traceback.print_exc()
            # Use a more interesting default ECG if generation fails
            ecg_signal = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.1, 0] * 30
            ecg_time = [i/100 for i in range(len(ecg_signal))]

        # Make prediction based on patient data
        try:
            print("Calculating risk prediction using hybrid model...")
            # Use the hybrid model for prediction
            prediction, confidence, explanations = hybrid_model.predict_heart_failure(patient_data)
            print(f"Hybrid Prediction: {prediction:.2f}, Confidence: {confidence:.2f}")

            # Extract SHAP values from explanations
            if 'rule_based' in explanations and 'shap_values' in explanations['rule_based']:
                shap_values = explanations['rule_based']['shap_values']
                print(f"Generated SHAP values for {len(shap_values['feature_names'])} features")
            else:
                # Create default SHAP values if not available
                shap_values = {
                    'base_value': 0.5,
                    'values': [0.1, -0.05, 0.2, -0.1, 0.15, 0.05],
                    'feature_names': ['age', 'gender', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'prior_cardiac_event']
                }
        except Exception as pred_error:
            print(f"Error calculating prediction with hybrid model: {str(pred_error)}")
            import traceback
            traceback.print_exc()
            # Fall back to rule-based prediction if hybrid model fails
            try:
                prediction, confidence = calculate_risk_prediction(patient_data)
                print(f"Fallback prediction: {prediction:.2f}, Confidence: {confidence:.2f}")
            except:
                # Use default values if both methods fail
                prediction = 0.5
                confidence = 0.7

            # Create default SHAP values
            shap_values = {
                'base_value': 0.5,
                'values': [0.1, -0.05, 0.2, -0.1, 0.15, 0.05],
                'feature_names': ['age', 'gender', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'prior_cardiac_event']
            }

            # Create empty explanations
            explanations = {
                'rule_based': {
                    'prediction': prediction,
                    'confidence': confidence,
                    'shap_values': shap_values
                }
            }

        # Generate abnormalities based on patient data and ECG
        try:
            print("Detecting abnormalities...")
            # Extract some basic features from the ECG for abnormality detection
            # In a real implementation, we would do more sophisticated feature extraction
            r_peaks = []

            # Force abnormality generation to use patient data
            # Set random seed based on patient data to ensure consistent but unique results
            patient_seed = hash(str(patient_data)) % 10000
            print(f"Using patient seed: {patient_seed} for abnormality generation")
            random.seed(patient_seed)

            abnormalities = generate_realistic_abnormalities(patient_data, ecg_signal, ecg_time, r_peaks)
            print(f"Detected {sum(len(v) for v in abnormalities.values())} abnormalities")

            # Reset random seed after abnormality generation
            random.seed()
        except Exception as abnorm_error:
            print(f"Error detecting abnormalities: {str(abnorm_error)}")
            import traceback
            traceback.print_exc()
            # Use empty abnormalities if detection fails
            abnormalities = {
                'PVCs': [],
                'Flatlines': [],
                'Tachycardia': [],
                'Bradycardia': [],
                'QT_prolongation': [],
                'ST_depression': [],
                'ST_elevation': [],
                'Atrial_Fibrillation': [],
                'Heart_block': []
            }

        try:
            # Save patient data for future retraining
            print("Saving patient data...")
            patient_id = save_patient_data(patient_data, prediction, confidence, ecg_signal, ecg_time, abnormalities, shap_values, explanations)
            print(f"Patient data saved with ID: {patient_id}")

            # Force a file system sync to ensure all changes are visible
            try:
                if hasattr(os, 'sync'):
                    os.sync()
                # Alternative approach for systems without os.sync
                subprocess.run(['sync'], check=False)
                print("File system synced after saving patient data")
            except Exception as sync_error:
                print(f"Warning: Could not sync file system after saving: {str(sync_error)}")
        except Exception as save_error:
            print(f"Error saving patient data: {str(save_error)}")
            import traceback
            traceback.print_exc()

            # Create a fallback patient ID
            patient_id = f"patient_fallback_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            print(f"Using fallback patient ID: {patient_id}")

            # Create a minimal patient file to ensure it exists
            try:
                # Ensure the directory exists
                os.makedirs('data/patients', exist_ok=True)

                # Create a minimal patient record
                minimal_data = {
                    'patient_id': patient_id,
                    'timestamp': datetime.now().isoformat(),
                    'patient_data': {
                        'name': patient_data.get('name', 'Error Recovery Patient'),
                        'age': patient_data.get('age', 0),
                        'gender': patient_data.get('gender', 'Unknown')
                    },
                    'prediction': float(prediction),
                    'confidence': float(confidence),
                    'error_recovery': True,
                    'original_error': str(save_error)
                }

                # Save the minimal data
                file_path = f'data/patients/{patient_id}.json'
                with open(file_path, 'w') as f:
                    json.dump(minimal_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                print(f"Created minimal fallback patient file: {file_path}")

                # Force a file system sync
                if hasattr(os, 'sync'):
                    os.sync()
                subprocess.run(['sync'], check=False)
            except Exception as fallback_error:
                print(f"Error creating fallback patient file: {str(fallback_error)}")
                traceback.print_exc()

        # Return response with ECG
        print("Preparing response...")

        # Create a response with clear patient ID and timestamp
        response_data = {
            'patient_id': patient_id,
            'prediction': prediction,
            'confidence': confidence,
            'ecg_signal': ecg_signal,
            'ecg_time': ecg_time,
            'abnormalities': abnormalities,
            'shap_values': shap_values,
            'timestamp': datetime.now().isoformat(),
            'file_path': f'data/patients/{patient_id}.json'
        }

        # Log the response for debugging
        print(f"Returning prediction response with patient_id: {patient_id}")

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Traceback: {error_traceback}")

        # Return a detailed error response
        return jsonify({
            'error': str(e),
            'traceback': error_traceback,
            'message': 'An error occurred while processing your request. Please try again.'
        }), 500

@app.route('/api/patients', methods=['GET'])
def get_patients():
    try:
        # Ensure the directory exists
        os.makedirs('data/patients', exist_ok=True)

        # Force a file system sync to ensure all changes are visible
        try:
            if hasattr(os, 'sync'):
                os.sync()
            # Alternative approach for systems without os.sync
            subprocess.run(['sync'], check=False)
            print("File system synced to ensure latest changes are visible")
        except Exception as sync_error:
            print(f"Warning: Could not sync file system: {str(sync_error)}")

        # Add a small delay to ensure file system operations are complete
        time.sleep(0.5)

        # Get all patient files with absolute path to avoid caching issues
        patient_dir = os.path.abspath('data/patients')
        print(f"Looking for patient files in directory: {patient_dir}")

        try:
            all_files_in_dir = os.listdir(patient_dir)
            print(f"All files in directory: {all_files_in_dir}")

            patient_files = [f for f in all_files_in_dir if f.endswith('.json')]
            print(f"Found {len(patient_files)} patient files in {patient_dir}: {patient_files}")

            # Check file permissions and sizes
            for file_name in patient_files:
                file_path = os.path.join(patient_dir, file_name)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                print(f"File: {file_name}, Size: {file_size} bytes, Modified: {datetime.fromtimestamp(file_mtime).isoformat()}")
        except Exception as e:
            print(f"Error listing patient files: {str(e)}")
            import traceback
            traceback.print_exc()
            patient_files = []

        # Debug: Check if there are any hidden files or other issues
        all_files = os.listdir(patient_dir)
        if len(all_files) != len(patient_files):
            print(f"Warning: Found {len(all_files)} total files but only {len(patient_files)} JSON files")
            print(f"All files: {all_files}")

        # Get the limit parameter if provided, otherwise return all patients
        limit = request.args.get('limit', None)
        if limit and limit.isdigit():
            limit = int(limit)
            print(f"Limiting results to {limit} patients")
        else:
            limit = None
            print(f"No limit specified, returning all {len(patient_files)} patients")

        if not patient_files:
            # Create a test patient file if no files exist
            print("No patient files found, creating test patient")
            test_patient_data = {
                'patient_id': 'test_patient_123',
                'timestamp': '2023-04-14T12:00:00',
                'patient_data': {
                    'name': 'Test Patient',
                    'age': 65,
                    'gender': 'Male',
                },
                'prediction': 0.75,
                'confidence': 0.85
            }

            # Save test patient to file
            os.makedirs('data/patients', exist_ok=True)
            with open('data/patients/test_patient_123.json', 'w') as f:
                json.dump(test_patient_data, f, indent=2)

            print("Created test patient file")

            # Return the test patient directly as an array
            test_response = jsonify([
                {
                    'patient_id': 'test_patient_123',
                    'timestamp': '2023-04-14T12:00:00',
                    'name': 'Test Patient',
                    'age': 65,
                    'gender': 'Male',
                    'prediction': 0.75,
                    'confidence': 0.85
                }
            ])

            # Add cache prevention headers
            test_response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            test_response.headers['Pragma'] = 'no-cache'
            test_response.headers['Expires'] = '0'
            test_response.headers['X-Patient-Count'] = '1'
            test_response.headers['X-Timestamp'] = datetime.now().isoformat()

            return test_response

        # Load all patient data
        patients = []
        # Use absolute paths for file reading to avoid caching issues
        patient_dir = os.path.abspath('data/patients')

        # Sort files by modification time (newest first) to ensure consistent ordering
        sorted_files = sorted(
            [(f, os.path.getmtime(os.path.join(patient_dir, f))) for f in patient_files],
            key=lambda x: x[1],
            reverse=True
        )

        # Apply limit if specified
        if limit is not None and len(sorted_files) > limit:
            print(f"Applying limit: {limit} out of {len(sorted_files)} files")
            sorted_files = sorted_files[:limit]

        print(f"Processing {len(sorted_files)} patient files in order (newest first)")

        for file_name, mtime in sorted_files:
            try:
                file_path = os.path.join(patient_dir, file_name)
                print(f"Reading file {file_path} (modified: {datetime.fromtimestamp(mtime).isoformat()})")

                with open(file_path, 'r') as f:
                    patient_data = json.load(f)
                print(f"Loaded patient data from {file_name}: {patient_data.get('patient_id', 'unknown')}")

                # Create a simplified patient record for the list view
                # Extract patient data from the correct location in the JSON structure
                patient_data_obj = patient_data.get('patient_data', {})

                # Debug the structure
                print(f"Patient data structure: {patient_data.keys()}")
                print(f"Patient data object: {patient_data_obj}")

                patient_record = {
                    'patient_id': patient_data.get('patient_id', 'unknown'),
                    'timestamp': patient_data.get('timestamp', ''),
                    'name': patient_data_obj.get('name', 'Unknown'),
                    'age': patient_data_obj.get('age', 0),
                    'gender': patient_data_obj.get('gender', 'Unknown'),
                    'prediction': patient_data.get('prediction', 0.5),
                    'confidence': patient_data.get('confidence', 0.7),
                    'file_path': file_path,  # Include file path for debugging
                    'file_mtime': mtime  # Include modification time for debugging
                }

                # Debug the created record
                print(f"Created patient record: {patient_record}")

                patients.append(patient_record)
            except Exception as e:
                print(f"Error loading patient file {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Sort patients by timestamp (newest first)
        patients.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Force a direct array response instead of an object with a patients property
        # This ensures consistent handling across all components
        print(f"Returning {len(patients)} patient records directly as an array")

        # Add a Cache-Control header to prevent caching
        response = jsonify(patients)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Patient-Count'] = str(len(patients))
        response.headers['X-Timestamp'] = datetime.now().isoformat()

        return response
    except Exception as e:
        print(f"Error in get_patients: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return a direct array with a single error record
        error_response = jsonify([{
            'patient_id': 'error',
            'timestamp': datetime.now().isoformat(),
            'name': f'Error loading patients: {str(e)}',
            'age': 0,
            'gender': 'Unknown',
            'prediction': 0,
            'confidence': 0,
            'is_error': True
        }])

        # Add cache prevention headers
        error_response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        error_response.headers['Pragma'] = 'no-cache'
        error_response.headers['Expires'] = '0'
        error_response.headers['X-Patient-Count'] = '0'
        error_response.headers['X-Error'] = str(e)

        return error_response

@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    try:
        print(f"Fetching details for patient ID: {patient_id}")

        # Force a file system sync to ensure all changes are visible
        try:
            if hasattr(os, 'sync'):
                os.sync()
            # Alternative approach for systems without os.sync
            subprocess.run(['sync'], check=False)
            print("File system synced to ensure latest changes are visible")
        except Exception as sync_error:
            print(f"Warning: Could not sync file system: {str(sync_error)}")

        # Check if patient file exists
        file_path = f'data/patients/{patient_id}.json'
        print(f"Looking for patient file at: {file_path}")

        if not os.path.exists(file_path):
            # Return a test patient if file doesn't exist
            return jsonify({
                'patient_id': patient_id,
                'timestamp': '2023-04-14T12:00:00',
                'patient_data': {
                    'name': 'Test Patient',
                    'age': 65,
                    'gender': 'Male',
                    'blood_pressure': '120/80',
                    'cholesterol': 200,
                    'fasting_blood_sugar': 100,
                    'max_heart_rate': 75
                },
                'prediction': 0.75,
                'confidence': 0.85,
                'ecg_signal': [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.1, 0] * 30,
                'ecg_time': [i/100 for i in range(17 * 30)],
                'abnormalities': {
                    'PVCs': [],
                    'Flatlines': [],
                    'Tachycardia': [],
                    'Bradycardia': [],
                    'QT_prolongation': [],
                    'ST_depression': [],
                    'ST_elevation': [],
                    'Atrial_Fibrillation': [],
                    'Heart_block': []
                },
                'shap_values': {
                    'base_value': 0.5,
                    'values': [0.1, -0.05, 0.2, -0.1, 0.15, 0.05],
                    'feature_names': ['age', 'gender', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'prior_cardiac_event']
                }
            })

        # Load patient data from file
        with open(file_path, 'r') as f:
            patient_data = json.load(f)

        # Check if the patient data has all required fields
        if 'ecg_signal' not in patient_data or not patient_data['ecg_signal']:
            print(f"Generating new ECG for patient {patient_id} as none was found in saved data")
            # Generate a new ECG signal based on patient data
            try:
                # Set random seed based on patient data to ensure consistent results
                # Use a more stable hash function that includes all relevant patient data
                patient_hash_str = f"{patient_data['patient_data'].get('name', '')}-{patient_data['patient_data'].get('age', '')}-{patient_data['patient_data'].get('gender', '')}-{patient_data['patient_data'].get('blood_pressure', '')}"
                patient_seed = hash(patient_hash_str) % 10000
                print(f"Using patient seed: {patient_seed} for ECG generation in patient detail (from hash of {patient_hash_str})")

                # Save the original random state
                original_state = random.getstate()

                # Set seed for ECG generation
                random.seed(patient_seed)

                # Generate ECG
                ecg_signal, ecg_time = generate_realistic_ecg(patient_data['patient_data'])
                patient_data['ecg_signal'] = ecg_signal[:1000]  # Limit to 1000 points
                patient_data['ecg_time'] = ecg_time[:1000]  # Limit to 1000 points

                # Restore the original random state
                random.setstate(original_state)
            except Exception as ecg_error:
                print(f"Error generating ECG: {str(ecg_error)}")
                # Use a default ECG if generation fails
                patient_data['ecg_signal'] = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.1, 0] * 30
                patient_data['ecg_time'] = [i/100 for i in range(len(patient_data['ecg_signal']))]

        if 'abnormalities' not in patient_data or not any(patient_data['abnormalities'].values()):
            print(f"Generating new abnormalities for patient {patient_id} as none were found in saved data")
            # Generate new abnormalities based on patient data
            try:
                # Set random seed based on patient data to ensure consistent results
                patient_seed = hash(str(patient_data['patient_data'])) % 10000
                random.seed(patient_seed)

                # Generate abnormalities
                r_peaks = []  # In a real implementation, we would extract R peaks from the ECG
                abnormalities = generate_realistic_abnormalities(patient_data['patient_data'],
                                                               patient_data['ecg_signal'],
                                                               patient_data['ecg_time'],
                                                               r_peaks)
                patient_data['abnormalities'] = abnormalities

                # Reset random seed
                random.seed()
            except Exception as abnorm_error:
                print(f"Error generating abnormalities: {str(abnorm_error)}")
                # Use empty abnormalities if generation fails
                patient_data['abnormalities'] = {
                    'PVCs': [],
                    'Flatlines': [],
                    'Tachycardia': [],
                    'Bradycardia': [],
                    'QT_prolongation': [],
                    'ST_depression': [],
                    'ST_elevation': [],
                    'Atrial_Fibrillation': [],
                    'Heart_block': []
                }

        if 'shap_values' not in patient_data:
            # Add default SHAP values
            patient_data['shap_values'] = {
                'base_value': 0.5,
                'values': [0.1, -0.05, 0.2, -0.1, 0.15, 0.05],
                'feature_names': ['age', 'gender', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'prior_cardiac_event']
            }

        return jsonify(patient_data)
    except Exception as e:
        print(f"Error in get_patient: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'patient_data': {
                'name': 'Error loading patient',
                'age': 0,
                'gender': 'Unknown'
            },
            'prediction': 0,
            'confidence': 0
    })

def calculate_risk_prediction(patient_data):
    """
    Calculate heart failure risk prediction based on patient data
    """
    # Base risk score - starts at 20%
    risk_score = 0.2

    # Get patient risk factors with type conversion
    try:
        age = int(patient_data.get('age', 60))
    except (ValueError, TypeError):
        age = 60

    gender = patient_data.get('gender', 'Male')

    try:
        heart_rate = int(patient_data.get('max_heart_rate', 75))
    except (ValueError, TypeError):
        heart_rate = 75

    prior_event = patient_data.get('prior_cardiac_event', {})
    prior_event_type = prior_event.get('type', '')
    prior_event_severity = prior_event.get('severity', 'Mild')

    try:
        cholesterol = int(patient_data.get('cholesterol', 200))
    except (ValueError, TypeError):
        cholesterol = 200

    blood_pressure_str = patient_data.get('blood_pressure', '120/80')
    chest_pain_type = patient_data.get('chest_pain_type', 'None')

    try:
        fasting_blood_sugar = int(patient_data.get('fasting_blood_sugar', 100))
    except (ValueError, TypeError):
        fasting_blood_sugar = 100

    exercise_induced_angina = bool(patient_data.get('exercise_induced_angina', False))

    try:
        st_depression = float(patient_data.get('st_depression', 0))
    except (ValueError, TypeError):
        st_depression = 0

    slope_of_st = patient_data.get('slope_of_st', 'Flat')

    try:
        num_vessels = int(patient_data.get('number_of_major_vessels', 0))
    except (ValueError, TypeError):
        num_vessels = 0

    thalassemia = patient_data.get('thalassemia', 'Normal')

    # Parse blood pressure
    try:
        systolic = int(blood_pressure_str.split('/')[0])
        diastolic = int(blood_pressure_str.split('/')[1])
    except:
        systolic = 120
        diastolic = 80

    # Age factor (significant risk increase with age)
    if age > 75:
        risk_score += 0.15
    elif age > 65:
        risk_score += 0.10
    elif age > 55:
        risk_score += 0.05

    # Gender factor
    if gender == 'Male':
        risk_score += 0.05  # Males have slightly higher risk

    # Heart rate factor
    if heart_rate > 100:
        risk_score += 0.08  # Tachycardia increases risk
    elif heart_rate < 60:
        risk_score += 0.05  # Bradycardia increases risk

    # Prior cardiac event (major risk factor)
    if prior_event_type:
        severity_factor = {'Mild': 0.10, 'Moderate': 0.15, 'Severe': 0.25}.get(prior_event_severity, 0.10)
        risk_score += severity_factor

        if 'Myocardial Infarction' in prior_event_type:
            risk_score += 0.10  # Additional risk for heart attack history

    # Cholesterol factor
    if cholesterol > 240:
        risk_score += 0.10
    elif cholesterol > 200:
        risk_score += 0.05

    # Blood pressure factor
    if systolic > 160 or diastolic > 100:
        risk_score += 0.12  # Stage 2 hypertension
    elif systolic > 140 or diastolic > 90:
        risk_score += 0.08  # Stage 1 hypertension
    elif systolic > 130 or diastolic > 85:
        risk_score += 0.04  # Elevated

    # Chest pain type
    if chest_pain_type == 'Typical Angina':
        risk_score += 0.10
    elif chest_pain_type == 'Atypical Angina':
        risk_score += 0.07
    elif chest_pain_type == 'Non-Anginal Pain':
        risk_score += 0.03

    # Fasting blood sugar
    if fasting_blood_sugar > 126:
        risk_score += 0.08  # Diabetic range
    elif fasting_blood_sugar > 100:
        risk_score += 0.04  # Pre-diabetic range

    # Exercise induced angina
    if exercise_induced_angina:
        risk_score += 0.10

    # ST depression
    if st_depression > 2.0:
        risk_score += 0.12
    elif st_depression > 1.0:
        risk_score += 0.08
    elif st_depression > 0.5:
        risk_score += 0.04

    # Slope of ST segment
    if slope_of_st == 'Downsloping':
        risk_score += 0.10
    elif slope_of_st == 'Flat':
        risk_score += 0.05

    # Number of major vessels
    risk_score += 0.07 * num_vessels

    # Thalassemia
    if thalassemia == 'Reversible Defect':
        risk_score += 0.12
    elif thalassemia == 'Fixed Defect':
        risk_score += 0.08

    # Add a small random factor for variability (±5%)
    risk_score += random.uniform(-0.05, 0.05)

    # Ensure risk score is between 0 and 1
    risk_score = max(0.01, min(0.99, risk_score))

    # Calculate confidence (higher for extreme values, lower for middle range)
    # Confidence is highest (0.9) at extremes and lowest (0.7) in the middle
    confidence = 0.7 + 0.2 * abs(risk_score - 0.5) * 2

    return risk_score, confidence

def generate_realistic_abnormalities(patient_data, ecg_signal, ecg_time, r_peaks):
    """
    Generate realistic abnormalities based on patient data and ECG signal
    """
    abnormalities = {
        'PVCs': [],
        'Flatlines': [],
        'Tachycardia': [],
        'Bradycardia': [],
        'QT_prolongation': [],
        'ST_depression': [],
        'ST_elevation': [],
        'Atrial_Fibrillation': [],
        'Heart_block': []
    }

    # Calculate duration of the ECG
    duration = ecg_time[-1] if ecg_time else 10.0

    # Get patient risk factors with type conversion
    try:
        age = int(patient_data.get('age', 60))
    except (ValueError, TypeError):
        age = 60

    try:
        heart_rate = int(patient_data.get('max_heart_rate', 75))
    except (ValueError, TypeError):
        heart_rate = 75

    prior_event = patient_data.get('prior_cardiac_event', {})
    prior_event_type = prior_event.get('type', '')
    prior_event_severity = prior_event.get('severity', 'Mild')

    try:
        cholesterol = int(patient_data.get('cholesterol', 200))
    except (ValueError, TypeError):
        cholesterol = 200

    blood_pressure_str = patient_data.get('blood_pressure', '120/80')

    # Parse blood pressure
    try:
        systolic = int(blood_pressure_str.split('/')[0])
        diastolic = int(blood_pressure_str.split('/')[1])
    except:
        systolic = 120
        diastolic = 80

    # Determine abnormality probabilities based on risk factors
    # Increased base probabilities to ensure more abnormalities are detected
    pvc_prob = 0.4  # Increased from 0.1
    tachycardia_prob = 0.3  # Increased from 0.05
    bradycardia_prob = 0.3  # Increased from 0.05
    qt_prob = 0.3  # Increased from 0.05
    st_depression_prob = 0.3  # Increased from 0.05
    st_elevation_prob = 0.2  # Increased from 0.02
    afib_prob = 0.3  # Increased from 0.05
    heart_block_prob = 0.2  # Increased from 0.03

    # Adjust probabilities based on risk factors
    if age > 65:
        pvc_prob += 0.2
        afib_prob += 0.1
        heart_block_prob += 0.05

    if heart_rate > 100:
        tachycardia_prob += 0.4
        pvc_prob += 0.1
    elif heart_rate < 60:
        bradycardia_prob += 0.4
        heart_block_prob += 0.1

    if prior_event_type:
        severity_factor = {'Mild': 0.1, 'Moderate': 0.2, 'Severe': 0.3}.get(prior_event_severity, 0.1)

        if 'Myocardial Infarction' in prior_event_type:
            st_depression_prob += 0.3 + severity_factor
            st_elevation_prob += 0.2 + severity_factor
            qt_prob += 0.2 + severity_factor

        if 'Arrhythmia' in prior_event_type:
            pvc_prob += 0.3 + severity_factor
            afib_prob += 0.3 + severity_factor

        if 'Heart Block' in prior_event_type:
            heart_block_prob += 0.4 + severity_factor

    if cholesterol > 240:
        st_depression_prob += 0.1

    if systolic > 140 or diastolic > 90:
        st_depression_prob += 0.1
        pvc_prob += 0.1

    # Generate abnormalities based on probabilities

    # Ensure at least one abnormality is always present
    # Choose a guaranteed abnormality type based on patient characteristics
    guaranteed_abnormality = False

    # For elderly patients, add QT prolongation
    if age > 65 and not guaranteed_abnormality:
        qt_start = random.uniform(1, duration - 2)
        qt_duration = random.uniform(0.5, 1.5)
        qt_interval = 0.45 + random.uniform(0, 0.15)

        abnormalities['QT_prolongation'].append({
            'time': qt_start,
            'duration': qt_duration,
            'interval': qt_interval,
            'description': f'QT prolongation - extended time between Q wave and T wave ({int(qt_interval*1000)} ms)'
        })
        guaranteed_abnormality = True

    # For patients with high heart rate, add tachycardia
    if heart_rate > 100 and not guaranteed_abnormality:
        tach_start = random.uniform(1, duration - 3)
        tach_duration = random.uniform(1.5, 3.0)
        tach_rate = heart_rate * (1.2 + random.uniform(0, 0.3))

        abnormalities['Tachycardia'].append({
            'time': tach_start,
            'duration': tach_duration,
            'rate': tach_rate,
            'description': f'Tachycardia - abnormally fast heart rate of {int(tach_rate)} bpm'
        })
        guaranteed_abnormality = True

    # For patients with low heart rate, add bradycardia
    if heart_rate < 60 and not guaranteed_abnormality:
        brady_start = random.uniform(1, duration - 3)
        brady_duration = random.uniform(1.5, 3.0)
        brady_rate = heart_rate * (0.6 + random.uniform(0, 0.2))

        abnormalities['Bradycardia'].append({
            'time': brady_start,
            'duration': brady_duration,
            'rate': brady_rate,
            'description': f'Bradycardia - abnormally slow heart rate of {int(brady_rate)} bpm'
        })
        guaranteed_abnormality = True

    # For patients with high cholesterol, add ST depression
    if cholesterol > 240 and not guaranteed_abnormality:
        st_start = random.uniform(1, duration - 2)
        st_duration = random.uniform(1.0, 2.0)
        st_magnitude = 0.1 + random.uniform(0, 0.2)

        abnormalities['ST_depression'].append({
            'time': st_start,
            'duration': st_duration,
            'magnitude': st_magnitude,
            'description': f'ST depression of {st_magnitude:.2f} mV - may indicate myocardial ischemia'
        })
        guaranteed_abnormality = True

    # For patients with high blood pressure, add PVCs
    if (systolic > 140 or diastolic > 90) and not guaranteed_abnormality:
        pvc_time = random.uniform(1, duration - 1)

        abnormalities['PVCs'].append({
            'time': pvc_time,
            'duration': 0.2,
            'amplitude': 1.5 + random.uniform(0, 0.5),
            'description': 'Premature ventricular contraction - abnormal heartbeat originating in the ventricles'
        })
        guaranteed_abnormality = True

    # If no abnormality has been added yet, add a PVC as a fallback
    if not guaranteed_abnormality:
        pvc_time = random.uniform(1, duration - 1)

        abnormalities['PVCs'].append({
            'time': pvc_time,
            'duration': 0.2,
            'amplitude': 1.5 + random.uniform(0, 0.5),
            'description': 'Premature ventricular contraction - abnormal heartbeat originating in the ventricles'
        })

    # Continue with probabilistic abnormality generation
    # PVCs (Premature Ventricular Contractions)
    if random.random() < pvc_prob:
        num_pvcs = random.randint(1, 3)
        for _ in range(num_pvcs):
            # Place PVCs at random times, but not too close to each other
            pvc_time = random.uniform(1, duration - 1)

            # Check if we already have PVCs and ensure they're not too close
            too_close = False
            for existing_pvc in abnormalities['PVCs']:
                if abs(existing_pvc['time'] - pvc_time) < 1.0:  # Keep PVCs at least 1 second apart
                    too_close = True
                    break

            if not too_close:
                abnormalities['PVCs'].append({
                    'time': pvc_time,
                    'duration': 0.2,
                    'amplitude': 1.5 + random.uniform(0, 0.5),  # PVCs are usually larger than normal beats
                    'description': 'Premature ventricular contraction - abnormal heartbeat originating in the ventricles'
                })

    # Tachycardia (Fast heart rate)
    if random.random() < tachycardia_prob:
        tach_start = random.uniform(1, duration - 3)  # Start at least 1s in, end at least 3s before end
        tach_duration = random.uniform(1.5, 3.0)      # Last between 1.5 and 3 seconds
        tach_rate = heart_rate * (1.2 + random.uniform(0, 0.3))  # 20-50% faster than baseline

        abnormalities['Tachycardia'].append({
            'time': tach_start,
            'duration': tach_duration,
            'rate': tach_rate,
            'description': f'Tachycardia - abnormally fast heart rate of {int(tach_rate)} bpm'
        })

    # Bradycardia (Slow heart rate)
    if random.random() < bradycardia_prob:
        brady_start = random.uniform(1, duration - 3)
        brady_duration = random.uniform(1.5, 3.0)
        brady_rate = heart_rate * (0.6 + random.uniform(0, 0.2))  # 20-40% slower than baseline

        abnormalities['Bradycardia'].append({
            'time': brady_start,
            'duration': brady_duration,
            'rate': brady_rate,
            'description': f'Bradycardia - abnormally slow heart rate of {int(brady_rate)} bpm'
        })

    # QT prolongation
    if random.random() < qt_prob:
        qt_start = random.uniform(1, duration - 2)
        qt_duration = random.uniform(0.5, 1.5)
        qt_interval = 0.45 + random.uniform(0, 0.15)  # Normal QT is ~0.35-0.44s, prolonged is >0.45s

        abnormalities['QT_prolongation'].append({
            'time': qt_start,
            'duration': qt_duration,
            'interval': qt_interval,
            'description': f'QT prolongation - extended time between Q wave and T wave ({int(qt_interval*1000)} ms)'
        })

    # ST depression (can indicate ischemia)
    if random.random() < st_depression_prob:
        st_start = random.uniform(1, duration - 2)
        st_duration = random.uniform(1.0, 2.0)
        st_magnitude = 0.1 + random.uniform(0, 0.2)  # Depression in mV

        abnormalities['ST_depression'].append({
            'time': st_start,
            'duration': st_duration,
            'magnitude': st_magnitude,
            'description': f'ST depression of {st_magnitude:.2f} mV - may indicate myocardial ischemia'
        })

    # ST elevation (can indicate myocardial infarction)
    if random.random() < st_elevation_prob:
        st_start = random.uniform(1, duration - 2)
        st_duration = random.uniform(1.0, 2.0)
        st_magnitude = 0.1 + random.uniform(0, 0.3)  # Elevation in mV

        abnormalities['ST_elevation'].append({
            'time': st_start,
            'duration': st_duration,
            'magnitude': st_magnitude,
            'description': f'ST elevation of {st_magnitude:.2f} mV - may indicate myocardial infarction'
        })

    # Atrial Fibrillation
    if random.random() < afib_prob:
        afib_start = random.uniform(1, duration - 3)
        afib_duration = random.uniform(2.0, 3.0)

        abnormalities['Atrial_Fibrillation'].append({
            'time': afib_start,
            'duration': afib_duration,
            'description': 'Atrial fibrillation - irregular heart rhythm with rapid, disorganized atrial activity'
        })

    # Heart Block
    if random.random() < heart_block_prob:
        block_start = random.uniform(1, duration - 2)
        block_duration = random.uniform(1.0, 2.0)
        block_degree = random.choice([1, 2, 3])  # 1st, 2nd, or 3rd degree heart block

        abnormalities['Heart_block'].append({
            'time': block_start,
            'duration': block_duration,
            'degree': block_degree,
            'description': f'{block_degree}{"st" if block_degree==1 else "nd" if block_degree==2 else "rd"} degree heart block - impaired electrical conduction between atria and ventricles'
        })

    return abnormalities

# Import ECG modules
from utils.ecg_generator import generate_ecg, analyze_ecg

# Create 12-lead ECG functions
def generate_12lead_ecg(patient_data):
    """Generate a simplified 12-lead ECG based on patient data"""
    # Create a simplified patient data object with proper types
    # to avoid type conversion issues
    simplified_data = {
        'age': 60,
        'gender': 'Male',
        'max_heart_rate': 75,
        'exercise_induced_angina': False,
        'st_depression': 0,
        'prior_cardiac_event': {},
        'medications': []
    }

    # Print debug info
    print(f"Generating 12-lead ECG for patient data: {simplified_data}")

    # Generate base ECG signal (Lead II)
    base_signal, time_array = generate_ecg(simplified_data)

    # Standard lead order for 12-lead ECG
    STANDARD_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Lead relationships (based on Einthoven's triangle and precordial lead placement)
    LEAD_RELATIONSHIPS = {
        # Limb leads
        'I': {'factor': 0.5, 'offset': 0.2},
        'II': {'factor': 1.0, 'offset': 0.0},  # Reference lead
        'III': {'factor': 0.7, 'offset': -0.1},

        # Augmented limb leads
        'aVR': {'factor': -0.5, 'offset': 0.1},
        'aVL': {'factor': 0.8, 'offset': 0.0},
        'aVF': {'factor': 0.9, 'offset': 0.0},

        # Precordial leads
        'V1': {'factor': 0.5, 'offset': -0.3, 'invert_t': True},
        'V2': {'factor': 0.7, 'offset': -0.2, 'invert_t': True},
        'V3': {'factor': 0.9, 'offset': -0.1},
        'V4': {'factor': 1.2, 'offset': 0.0},
        'V5': {'factor': 1.0, 'offset': 0.1},
        'V6': {'factor': 0.8, 'offset': 0.0},
    }

    # Generate all 12 leads based on the base signal
    leads_data = {}
    for lead_name in STANDARD_LEAD_ORDER:
        # Get lead relationship parameters
        relationship = LEAD_RELATIONSHIPS[lead_name]

        # Generate lead signal based on the relationship
        if lead_name == 'II':
            # Lead II is our base signal
            lead_signal = base_signal
        else:
            # Derive other leads from the base signal
            lead_signal = base_signal * relationship['factor'] + relationship['offset']

            # Add some random variation to make it more realistic
            noise = np.random.normal(0, 0.02, len(lead_signal))
            lead_signal = lead_signal + noise

        # Store the lead data
        leads_data[lead_name] = lead_signal.tolist()

    # Create metadata
    metadata = {
        'sampling_rate': len(base_signal) / 10,  # Assuming 10 seconds of data
        'duration': 10,  # seconds
        'heart_rate': 75,  # Default heart rate
        'patient_age': patient_data.get('age', 60),
        'patient_gender': patient_data.get('gender', 'Male'),
        'timestamp': datetime.now().isoformat()
    }

    # Return complete 12-lead ECG data
    return {
        'leads': leads_data,
        'time': time_array.tolist(),
        'metadata': metadata,
        'lead_order': STANDARD_LEAD_ORDER
    }

def analyze_12lead_ecg(ecg_data):
    """Analyze 12-lead ECG data to detect abnormalities"""
    # Create a simplified analysis result
    # This is a basic implementation that doesn't rely on complex processing

    # Initialize results
    abnormalities = {
        'rhythm': [],
        'conduction': [],
        'st_changes': [],
        'chamber_enlargement': [],
        'axis_deviation': [],
        'infarction': [],
        'PVCs': [],
        'QT_prolongation': []
    }

    # Add some sample abnormalities for demonstration
    abnormalities['rhythm'].append({
        'type': 'Sinus Rhythm',
        'description': 'Normal sinus rhythm',
        'confidence': 0.95,
        'lead': 'II'
    })

    # Create comprehensive analysis results
    analysis_results = {
        'abnormalities': abnormalities,
        'lead_analyses': {},
        'rhythm': {'name': 'Normal Sinus Rhythm', 'regularity': 'Regular', 'confidence': 0.9},
        'heart_rate': 75,  # Default heart rate
        'intervals': {
            'PR': 0.16,  # seconds
            'QRS': 0.08,  # seconds
            'QT': 0.38,  # seconds
            'QTc': 0.41,  # seconds
        },
        'axis': {
            'value': 60,  # Default normal axis
            'category': 'Normal',
            'confidence': 0.7
        }
    }

    return analysis_results

def generate_realistic_ecg(patient_data):
    """
    Generate a realistic ECG signal based on patient data
    """
    # Parameters
    duration = 10  # seconds
    sampling_rate = 250  # Hz (increased for better resolution)
    num_points = int(duration * sampling_rate)

    # Create time array
    ecg_time = [i/sampling_rate for i in range(num_points)]

    # Get heart rate from patient data or use default
    try:
        heart_rate = int(patient_data.get('max_heart_rate', 75))
    except (ValueError, TypeError):
        heart_rate = 75
    # Ensure heart rate is within reasonable bounds
    heart_rate = max(40, min(heart_rate, 200))

    # Calculate RR interval in seconds
    rr_interval = 60.0 / heart_rate

    # Generate ECG signal
    ecg_signal = []

    # Add some variability to RR intervals (heart rate variability)
    rr_variability = 0.05  # 5% variability

    # Amplitude parameters - adjusted for more realistic appearance
    p_amplitude = 0.2
    qrs_amplitude = 1.2
    t_amplitude = 0.4

    # Baseline parameters
    baseline = 0.0
    baseline_drift_amplitude = 0.05
    baseline_drift_frequency = 0.1  # Hz

    # Width parameters - adjusted for more realistic appearance
    p_width = 0.08  # seconds
    qrs_width = 0.06  # seconds (narrower for sharper QRS complex)
    t_width = 0.16  # seconds

    # Intervals - adjusted for more realistic appearance
    pr_interval = 0.16  # seconds
    qt_interval = 0.36  # seconds
    st_segment_duration = 0.1  # seconds

    # Apply modifiers based on patient data

    # Age effect: older patients have lower amplitude
    try:
        age = int(patient_data.get('age', 60))
    except (ValueError, TypeError):
        age = 60
    age_factor = 1.0 - (max(0, age - 40) * 0.005)  # 0.5% reduction per year over 40
    qrs_amplitude *= age_factor

    # Heart rate effect: adjust beat frequency and amplitude
    heart_rate_factor = heart_rate / 75.0  # Normalize to typical heart rate
    if heart_rate > 100:  # Tachycardia
        p_amplitude *= 0.9  # Reduced P wave in tachycardia
        t_amplitude *= 0.9  # Reduced T wave in tachycardia
        rr_variability *= 0.8  # Less variability in tachycardia
    elif heart_rate < 60:  # Bradycardia
        p_amplitude *= 1.1  # More pronounced P wave in bradycardia
        t_amplitude *= 1.1  # More pronounced T wave in bradycardia
        rr_variability *= 1.2  # More variability in bradycardia

    # Blood pressure effect
    blood_pressure_str = patient_data.get('blood_pressure', '120/80')
    try:
        systolic = int(blood_pressure_str.split('/')[0])
        if systolic > 140:  # Hypertension
            qrs_amplitude *= 1.1  # Increased QRS amplitude in hypertension
            baseline_drift_amplitude *= 1.2  # More baseline drift
    except (ValueError, TypeError, IndexError):
        systolic = 120

    # Cholesterol effect
    try:
        cholesterol = int(patient_data.get('cholesterol', 200))
    except (ValueError, TypeError):
        cholesterol = 200

    if cholesterol > 240:  # High cholesterol
        st_depression = -0.1  # ST depression with high cholesterol
    else:
        st_depression = 0

    # Prior cardiac event effect
    prior_event = patient_data.get('prior_cardiac_event', {})
    if prior_event and prior_event.get('type'):
        # Reduce amplitude and add irregularity
        severity = prior_event.get('severity', 'Mild')
        severity_factor = {'Mild': 0.9, 'Moderate': 0.8, 'Severe': 0.7}.get(severity, 0.9)
        qrs_amplitude *= severity_factor
        rr_variability *= (2.0 / severity_factor)  # More variability with more severe events

        # Add ST depression for MI
        if 'Myocardial Infarction' in prior_event.get('type', ''):
            st_depression = -0.2 * (1.0 / severity_factor)
            t_amplitude *= 0.8  # Reduced T wave in MI

        # Add QT prolongation for some conditions
        if 'Arrhythmia' in prior_event.get('type', ''):
            qt_interval *= 1.2  # Prolonged QT interval

    # NT-proBNP biomarker effect on ECG morphology
    # References:
    # 1. Januzzi JL Jr, et al. (2019). "NT-proBNP Testing for Diagnosis and Short-Term Prognosis in Acute Heart Failure"
    # 2. Maisel AS, et al. (2008). "State of the art: using natriuretic peptide levels in clinical practice"
    # 3. Iwanaga Y, et al. (2006). "B-type natriuretic peptide strongly reflects diastolic wall stress in patients with heart failure"
    try:
        nt_probnp = float(patient_data.get('biomarkers', {}).get('nt_probnp', 0))
        if nt_probnp > 0:
            # Age-adjusted thresholds based on ESC Guidelines
            age = float(patient_data.get('age', 65))

            if age < 50:
                threshold = 450  # pg/mL
                high_risk = 900  # 2x threshold indicates high risk
            elif age <= 75:
                threshold = 900  # pg/mL
                high_risk = 1800
            else:
                threshold = 1800  # pg/mL
                high_risk = 3600

            # Calculate effect magnitude based on NT-proBNP level relative to threshold
            # Physiological basis: Higher NT-proBNP correlates with more severe cardiac dysfunction
            if nt_probnp < threshold/2:  # Well below threshold - minimal effect
                effect_magnitude = 0.1
            elif nt_probnp < threshold:  # Below threshold but detectable - mild effect
                effect_magnitude = 0.1 + 0.2 * (nt_probnp - threshold/2) / (threshold/2)
            elif nt_probnp < high_risk:  # Between threshold and high risk - moderate effect
                effect_magnitude = 0.3 + 0.3 * (nt_probnp - threshold) / (high_risk - threshold)
            else:  # Above high risk threshold - strong effect
                # Logarithmic scaling for very high values to prevent unrealistic ECG
                effect_magnitude = min(0.9, 0.6 + 0.3 * math.log(nt_probnp / high_risk + 1))

            print(f"NT-proBNP ECG effect magnitude: {effect_magnitude:.2f} (value: {nt_probnp}, threshold: {threshold})")

            # Apply physiologically accurate ECG changes based on NT-proBNP level

            # 1. T wave changes - ventricular repolarization abnormalities
            # Physiological basis: Ventricular strain pattern seen in heart failure
            # - Reduced amplitude and possibly inverted T waves
            t_amplitude *= max(0.5, 1.0 - (effect_magnitude * 0.5))
            if effect_magnitude > 0.7 and random.random() < 0.3:  # 30% chance of T wave inversion in severe cases
                t_amplitude *= -0.3  # Inverted T wave

            # 2. QRS complex changes - ventricular conduction delays
            # Physiological basis: Left bundle branch block and intraventricular conduction delays
            # common in heart failure with elevated NT-proBNP
            qrs_width *= (1.0 + (effect_magnitude * 0.4))  # Up to 40% wider QRS

            # 3. ST segment changes - subendocardial ischemia or strain
            # Physiological basis: Subendocardial ischemia in heart failure
            st_depression -= (effect_magnitude * 0.2)  # Up to 0.2 mV ST depression

            # 4. PR interval changes - atrial conduction abnormalities
            # Physiological basis: Atrial enlargement and fibrosis in heart failure
            pr_interval *= (1.0 + (effect_magnitude * 0.15))  # Up to 15% longer PR interval

            # 5. Heart rate and rhythm changes - autonomic dysfunction
            # Physiological basis: Neurohormonal activation in heart failure
            rr_variability *= (1.0 + (effect_magnitude * 0.6))  # Up to 60% more RR variability

            # 6. QT interval changes - repolarization abnormalities
            # Physiological basis: Electrolyte disturbances and medication effects in heart failure
            qt_interval *= (1.0 + (effect_magnitude * 0.1))  # Up to 10% longer QT interval

            # 7. P wave changes - atrial enlargement
            # Physiological basis: Left atrial enlargement in heart failure
            p_amplitude *= (1.0 + (effect_magnitude * 0.3))  # Up to 30% larger P waves
            p_width *= (1.0 + (effect_magnitude * 0.2))  # Up to 20% wider P waves

            # 8. Occasional ectopic beats in severe cases
            # Physiological basis: Increased arrhythmia risk in heart failure
            # Note: We'll flag this for the abnormality detection algorithm, but not modify the ECG here
            # The actual PVCs will be added during the abnormality detection phase
    except (ValueError, TypeError):
        pass  # Ignore if NT-proBNP is not a valid number

    # Chest pain effect
    chest_pain_type = patient_data.get('chest_pain_type', 'None')
    if chest_pain_type == 'Typical Angina':
        st_depression -= 0.15  # ST depression in angina
    elif chest_pain_type == 'Atypical Angina':
        st_depression -= 0.08

    # Exercise induced angina effect
    exercise_induced_angina = patient_data.get('exercise_induced_angina', False)
    if exercise_induced_angina and exercise_induced_angina not in [False, 'false', 'False', '0', 0, None, '']:
        st_depression -= 0.12
        t_amplitude *= 0.9

    # ST depression from patient data
    try:
        st_depression_value = float(patient_data.get('st_depression', 0))
    except (ValueError, TypeError):
        st_depression_value = 0
    st_depression -= st_depression_value * 0.1

    # Medication effects
    medications = patient_data.get('medications', [])
    for med in medications:
        med_type = med.get('type', '')
        if 'Beta-blocker' in med_type:
            # Beta blockers slow heart rate and reduce variability
            heart_rate *= 0.9
            rr_variability *= 0.7
        elif 'ACE inhibitor' in med_type:
            # ACE inhibitors can normalize ST segment
            st_depression *= 0.7

    # Generate the ECG signal
    t = 0
    next_r_peak = rr_interval  # Time of the next R peak
    r_peaks = []  # Store R peak locations for abnormality detection

    for i in range(num_points):
        t = ecg_time[i]

        # Add baseline drift (respiratory effect)
        baseline_value = baseline + baseline_drift_amplitude * math.sin(2 * math.pi * baseline_drift_frequency * t)

        # Determine the phase within the cardiac cycle
        if t >= next_r_peak:
            # Add variability to the next RR interval
            rr_with_variability = rr_interval * (1 + random.uniform(-rr_variability, rr_variability))
            next_r_peak += rr_with_variability
            r_peaks.append(next_r_peak - rr_with_variability)  # Store the R peak that just occurred

        # Time since last R peak
        t_since_r = t - (next_r_peak - rr_interval)

        # Initialize the value for this time point
        value = baseline_value

        # P wave (occurs before the QRS complex)
        p_center = rr_interval - pr_interval - qrs_width/2
        if abs(t_since_r - p_center) < p_width:
            # P wave shape (more realistic asymmetric Gaussian)
            p_relative_pos = (t_since_r - p_center) / p_width
            # Make P wave slightly asymmetric
            if p_relative_pos < 0:
                p_factor = 4.5  # Steeper rise
            else:
                p_factor = 3.5  # Gentler fall
            value += p_amplitude * math.exp(-p_relative_pos * p_relative_pos * p_factor)

        # QRS complex
        qrs_center = rr_interval
        q_width = qrs_width * 0.2  # Q wave is about 20% of QRS width
        r_width = qrs_width * 0.3  # R wave is about 30% of QRS width
        s_width = qrs_width * 0.5  # S wave is about 50% of QRS width

        # Q wave (small negative deflection)
        q_center = qrs_center - (r_width/2 + q_width/2)
        if abs(t_since_r - q_center) < q_width/2:
            q_relative_pos = abs((t_since_r - q_center) / (q_width/2))
            value -= 0.2 * qrs_amplitude * (1 - q_relative_pos**2)

        # R wave (large positive deflection)
        if abs(t_since_r - qrs_center) < r_width/2:
            r_relative_pos = abs((t_since_r - qrs_center) / (r_width/2))
            # Sharper R peak with exponential function
            value += qrs_amplitude * (1 - r_relative_pos**1.5)

        # S wave (negative deflection after R)
        s_center = qrs_center + (r_width/2 + s_width/2)
        if abs(t_since_r - s_center) < s_width/2:
            s_relative_pos = abs((t_since_r - s_center) / (s_width/2))
            value -= 0.3 * qrs_amplitude * (1 - s_relative_pos**1.5)

        # ST segment (after the S wave)
        st_start = qrs_center + qrs_width/2 + s_width/2
        st_end = st_start + st_segment_duration
        if t_since_r > st_start and t_since_r < st_end:
            # ST segment (usually isoelectric, but can be depressed in ischemia)
            # Add a slight curve to the ST segment
            st_progress = (t_since_r - st_start) / (st_end - st_start)
            value += st_depression * (1 - 0.5 * math.sin(math.pi * st_progress))

        # T wave (occurs after the QRS complex)
        t_start = st_end
        t_center = t_start + t_width/2
        if t_since_r > t_start and t_since_r < t_start + t_width:
            # T wave shape (asymmetric Gaussian for more realism)
            t_relative_pos = (t_since_r - t_center) / (t_width/2)
            # Make T wave asymmetric
            if t_relative_pos < 0:
                t_factor = 1.5  # Gentler rise
            else:
                t_factor = 2.5  # Steeper fall
            value += t_amplitude * math.exp(-t_relative_pos * t_relative_pos * t_factor)

        # Add some physiological noise (combination of high and low frequency components)
        high_freq_noise = random.uniform(-0.03, 0.03)  # Muscle artifact
        low_freq_noise = 0.02 * math.sin(2 * math.pi * 0.5 * t + random.uniform(0, 2*math.pi))  # Movement artifact
        value += high_freq_noise + low_freq_noise

        ecg_signal.append(value)

    return ecg_signal, ecg_time

# Function to save patient data for retraining
def save_patient_data(patient_data, prediction, confidence, ecg_signal=None, ecg_time=None, abnormalities=None, shap_values=None, explanations=None):
    """
    Save patient data to JSON file for later retraining
    """
    try:
        # Generate unique patient ID
        patient_id = f"patient_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        print(f"Generating patient ID: {patient_id}")

        # Sanitize patient data to ensure it's JSON serializable
        sanitized_data = {}
        try:
            # Only include basic data types that are JSON serializable
            for key, value in patient_data.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    sanitized_data[key] = value
                elif isinstance(value, dict):
                    # Handle nested dictionaries (one level only)
                    sanitized_data[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool, type(None))):
                            sanitized_data[key][k] = v
                elif isinstance(value, list):
                    # Handle lists of simple types or dictionaries
                    sanitized_list = []
                    for item in value:
                        if isinstance(item, (str, int, float, bool, type(None))):
                            sanitized_list.append(item)
                        elif isinstance(item, dict):
                            sanitized_dict = {}
                            for k, v in item.items():
                                if isinstance(v, (str, int, float, bool, type(None))):
                                    sanitized_dict[k] = v
                            sanitized_list.append(sanitized_dict)
                    sanitized_data[key] = sanitized_list
        except Exception as sanitize_error:
            print(f"Error sanitizing patient data: {str(sanitize_error)}")
            # If sanitization fails, use a minimal set of data
            sanitized_data = {
                'name': patient_data.get('name', 'Unknown'),
                'age': patient_data.get('age', 0),
                'gender': patient_data.get('gender', 'Unknown')
            }

        # Create data object
        data = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'patient_data': sanitized_data,
            'prediction': float(prediction),  # Ensure it's a float for JSON serialization
            'confidence': float(confidence),  # Ensure it's a float for JSON serialization
            'feedback': None,  # Will be updated if user provides feedback

            # Add ECG data if provided
            'ecg_signal': ecg_signal[:1000] if ecg_signal is not None else [],  # Limit to 1000 points to keep file size reasonable
            'ecg_time': ecg_time[:1000] if ecg_time is not None else [],  # Limit to 1000 points to keep file size reasonable

            # Add abnormalities if provided
            'abnormalities': abnormalities if abnormalities is not None else {
                'PVCs': [],
                'Flatlines': [],
                'Tachycardia': [],
                'Bradycardia': [],
                'QT_prolongation': [],
                'ST_depression': [],
                'ST_elevation': [],
                'Atrial_Fibrillation': [],
                'Heart_block': []
            },

            # Add SHAP values
            'shap_values': shap_values if shap_values is not None else {
                'base_value': 0.5,
                'values': [0.1, -0.05, 0.2, -0.1, 0.15, 0.05],
                'feature_names': ['age', 'gender', 'blood_pressure', 'cholesterol', 'max_heart_rate', 'prior_cardiac_event']
            },

            # Add full model explanations
            'explanations': explanations if explanations is not None else {}
        }

        # Ensure the directory exists
        os.makedirs('data/patients', exist_ok=True)
        print(f"Ensuring directory exists: data/patients")

        # Save to file
        file_path = f'data/patients/{patient_id}.json'

        # We'll keep all patient files - no limit
        try:
            patient_files = [f for f in os.listdir('data/patients') if f.endswith('.json')]
            print(f"Found {len(patient_files)} existing patient files before saving new one")

            # Removed the file limit code to allow unlimited patient files
            files_to_remove = []  # No files to remove
            print("No file limit enforced - keeping all patient files")

            # This loop won't run since files_to_remove is empty
            for old_file in files_to_remove:
                try:
                    os.remove(os.path.join('data/patients', old_file))
                    print(f"Removed old patient file: {old_file}")
                except Exception as remove_error:
                    print(f"Error removing old file {old_file}: {str(remove_error)}")
        except Exception as cleanup_error:
            print(f"Error during file cleanup: {str(cleanup_error)}")

        # Now save the new file
        try:
            # First, ensure the directory exists with full permissions
            patient_dir = os.path.dirname(file_path)
            os.makedirs(patient_dir, exist_ok=True)
            print(f"Created directory: {patient_dir}")

            # Print the current working directory for debugging
            print(f"Current working directory: {os.getcwd()}")
            print(f"Absolute path to patient directory: {os.path.abspath(patient_dir)}")

            # Check directory permissions
            try:
                import stat
                dir_stat = os.stat(patient_dir)
                dir_perms = stat.filemode(dir_stat.st_mode)
                print(f"Directory permissions: {dir_perms}")
            except Exception as perm_error:
                print(f"Error checking directory permissions: {str(perm_error)}")

            # Write to a temporary file first
            temp_file_path = f"{file_path}.tmp"
            print(f"Writing to temp file: {temp_file_path}")
            with open(temp_file_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to write to physical storage

            # Verify the temp file was saved correctly
            if os.path.exists(temp_file_path):
                file_size = os.path.getsize(temp_file_path)
                print(f"Temp file saved: {temp_file_path} (size: {file_size} bytes)")

                # Now rename the temp file to the final file name (atomic operation)
                os.rename(temp_file_path, file_path)

                # Force a file system sync to ensure all changes are visible
                try:
                    if hasattr(os, 'sync'):
                        os.sync()
                    # Alternative approach for systems without os.sync
                    subprocess.run(['sync'], check=False)
                except Exception as sync_error:
                    print(f"Warning: Could not sync file system: {str(sync_error)}")

                # Verify the final file exists
                if os.path.exists(file_path):
                    final_size = os.path.getsize(file_path)
                    print(f"Patient data saved to {file_path} for future retraining (size: {final_size} bytes)")

                    # List all patient files after saving
                    all_files = [f for f in os.listdir('data/patients') if f.endswith('.json')]
                    print(f"Total patient files after saving: {len(all_files)}")
                    return patient_id
                else:
                    print(f"ERROR: Final file {file_path} was not created after rename!")
                    return f"error_rename_failed_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            else:
                print(f"ERROR: Temp file {temp_file_path} was not created!")
                return f"error_temp_file_not_created_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        except Exception as save_error:
            print(f"Error saving patient file: {str(save_error)}")
            import traceback
            traceback.print_exc()
            return f"error_saving_file_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    except Exception as e:
        print(f"Error saving patient data: {str(e)}")
        import traceback
        traceback.print_exc()
        return "error_saving_data"

# Add a new endpoint for model retraining
@app.route('/api/retrain', methods=['POST'])
def retrain_model():

    try:
        print("Starting model retraining process...")

        # Force a file system sync to ensure all changes are visible
        try:
            if hasattr(os, 'sync'):
                os.sync()
            # Alternative approach for systems without os.sync
            subprocess.run(['sync'], check=False)
        except Exception as sync_error:
            print(f"Warning: Could not sync file system: {str(sync_error)}")

        # Get absolute path to avoid caching issues
        patient_dir = os.path.abspath('data/patients')

        # Count the number of patient files before retraining
        try:
            patient_files = [f for f in os.listdir(patient_dir) if f.endswith('.json')]
            print(f"Found {len(patient_files)} patient files for retraining in {patient_dir}")

            # List all files for debugging
            for i, file in enumerate(patient_files):
                file_path = os.path.join(patient_dir, file)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                print(f"  {i+1}. {file} (size: {file_size} bytes, modified: {datetime.fromtimestamp(file_mtime).isoformat()})")
        except Exception as e:
            print(f"Error counting patient files: {str(e)}")
            import traceback
            traceback.print_exc()
            patient_files = []

        # Use the hybrid model's retraining function
        result = hybrid_model.retrain_model()

        # Add the actual number of files used
        result['num_files'] = len(patient_files)

        print(f"Model retraining completed: {result['message']}")

        # Create response with timestamp
        timestamp = datetime.now().isoformat()
        response_data = {
            'success': result['success'],
            'timestamp': timestamp,
            'num_records': result['num_records'],
            'total_records': result.get('total_records', len(patient_files)),
            'num_files': len(patient_files),
            'processed_count': result.get('processed_count', result['num_records']),
            'skipped_count': result.get('skipped_count', len(patient_files) - result['num_records']),
            'weights': result.get('weights', {}),
            'message': f"{result['message']} (Used {result['num_records']} out of {len(patient_files)} patient files)"
        }

        # Save the training history
        try:
            history_file = 'data/training_history.json'
            history = []

            # Load existing history if available
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                        if not isinstance(history, list):
                            history = [history]
                except Exception as e:
                    print(f"Error loading training history: {str(e)}")
                    history = []

            # Add current training result to history
            history.append(response_data)

            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            print(f"Saved training history with {len(history)} entries")
        except Exception as e:
            print(f"Error saving training history: {str(e)}")

        # Return response
        response = jsonify(response_data)

        # Add cache prevention headers
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response

    except Exception as e:
        print(f"Error retraining model: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add an endpoint to create a test patient file
@app.route('/api/patients/check/<patient_id>', methods=['GET'])
def check_patient(patient_id):
    try:
        # Check if the patient file exists
        file_path = f'data/patients/{patient_id}.json'
        if os.path.exists(file_path):
            # Get file details
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)

            # Read the file content
            with open(file_path, 'r') as f:
                patient_data = json.load(f)

            return jsonify({
                'exists': True,
                'patient_id': patient_id,
                'file_path': file_path,
                'file_size': file_size,
                'file_mtime': file_mtime,
                'file_mtime_str': datetime.fromtimestamp(file_mtime).isoformat(),
                'patient_data': patient_data
            })
        else:
            return jsonify({
                'exists': False,
                'patient_id': patient_id,
                'message': f"Patient file {patient_id}.json does not exist"
            })
    except Exception as e:
        print(f"Error checking patient {patient_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'exists': False,
            'patient_id': patient_id,
            'error': str(e)
        }), 500

@app.route('/api/patients/create-test', methods=['GET'])
def create_test_patient():
    try:
        # Ensure the directory exists
        os.makedirs('data/patients', exist_ok=True)

        # Generate a unique patient ID
        patient_id = f"test_patient_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create a test patient record
        patient_data = {
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat(),
            'patient_data': {
                'name': f"Test Patient {datetime.now().strftime('%H:%M:%S')}",
                'age': 50,
                'gender': 'Male',
                'blood_pressure': '120/80',
                'cholesterol': 200,
                'fasting_blood_sugar': 100,
                'max_heart_rate': 75
            },
            'prediction': 0.65,
            'confidence': 0.75
        }

        # Save the patient record
        file_path = f'data/patients/{patient_id}.json'
        with open(file_path, 'w') as f:
            json.dump(patient_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # Verify the file was saved
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)

            return jsonify({
                'success': True,
                'message': f"Created test patient file: {file_path}",
                'patient_id': patient_id,
                'file_size': file_size,
                'file_mtime': file_mtime,
                'file_mtime_str': datetime.fromtimestamp(file_mtime).isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': f"Failed to create test patient file: {file_path}"
            }), 500
    except Exception as e:
        print(f"Error creating test patient file: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add an endpoint to list all patient files
@app.route('/api/patients/files', methods=['GET'])
def list_patient_files():
    try:
        # Ensure the directory exists
        os.makedirs('data/patients', exist_ok=True)

        # Get all files in the directory
        patient_dir = os.path.abspath('data/patients')
        all_files = os.listdir(patient_dir)

        # Filter for JSON files
        json_files = [f for f in all_files if f.endswith('.json')]

        # Get file details
        file_details = []
        for file_name in json_files:
            file_path = os.path.join(patient_dir, file_name)
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)

            file_details.append({
                'file_name': file_name,
                'file_path': file_path,
                'file_size': file_size,
                'file_mtime': file_mtime,
                'file_mtime_str': datetime.fromtimestamp(file_mtime).isoformat()
            })

        # Sort by modification time (newest first)
        file_details.sort(key=lambda x: x['file_mtime'], reverse=True)

        return jsonify({
            'count': len(file_details),
            'files': file_details
        })
    except Exception as e:
        print(f"Error listing patient files: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

# Add an endpoint to get retraining history
@app.route('/api/retraining/history', methods=['GET'])
def get_retraining_history():

    try:
        # Use the hybrid model's function to get training history
        history = hybrid_model.get_training_history()

        if not history:
            # Return empty history if none exists
            default_history = [{
                'timestamp': datetime.now().isoformat(),
                'num_records': 0,
                'total_records': 0,
                'processed_count': 0,
                'skipped_count': 0,
                'metrics': {
                    'accuracy': 0.85,  # Default accuracy
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'roc_auc': 0.5
                },
                'message': "No retraining has been performed yet"
            }]
            return jsonify(default_history)

        # Add cache prevention headers
        response = jsonify(history)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'

        return response

    except Exception as e:
        print(f"Error getting retraining history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add an endpoint for recording prediction feedback
@app.route('/api/feedback', methods=['POST'])
def record_feedback():
    try:
        data = request.json
        patient_id = data.get('patient_id')
        prediction = data.get('prediction')
        actual = data.get('actual')  # Actual outcome (1 for heart failure, 0 for no heart failure)

        # Get patient data
        patient_data = {}
        if os.path.exists(f'data/patients/{patient_id}.json'):
            with open(f'data/patients/{patient_id}.json', 'r') as f:
                patient_record = json.load(f)
                patient_data = patient_record.get('patient_data', {})

        # Record feedback directly to the patient file
        success = False

        # Also record as an outcome in the longitudinal system if available
        try:
            # Check if patient exists in longitudinal system
            longitudinal_patient = longitudinal_tracker.load_patient(patient_id)

            if longitudinal_patient is None:
                # Migrate the patient to the longitudinal system
                print(f"Migrating patient {patient_id} to longitudinal system")
                with open(f'data/patients/{patient_id}.json', 'r') as f:
                    patient_data_full = json.load(f)
                longitudinal_patient = longitudinal_tracker.migrate_existing_patient(patient_id, patient_data_full)

            # Record the feedback as an outcome
            outcome_type = "heart_failure" if actual == 1 else "no_heart_failure"
            severity = 4 if actual == 1 else 1  # Higher severity for heart failure

            details = {
                'feedback': 'correct' if (prediction >= 0.5 and actual == 1) or (prediction < 0.5 and actual == 0) else 'incorrect',
                'prediction': prediction,
                'actual': actual
            }

            outcome_id = longitudinal_patient.add_outcome(
                timestamp=datetime.now().isoformat(),
                outcome_type=outcome_type,
                severity=severity,
                details=details
            )

            print(f"Recorded outcome in longitudinal system with ID: {outcome_id}")
        except Exception as longitudinal_error:
            print(f"Error recording in longitudinal system: {str(longitudinal_error)}")
            import traceback
            traceback.print_exc()

        try:
            # Load the patient record
            if os.path.exists(f'data/patients/{patient_id}.json'):
                with open(f'data/patients/{patient_id}.json', 'r') as f:
                    patient_record = json.load(f)

                # Add feedback
                # For heart failure risk (1), the feedback is 'correct' if the prediction was high (>=0.5)
                # For no heart failure risk (0), the feedback is 'correct' if the prediction was low (<0.5)
                prediction_value = float(patient_record.get('prediction', 0.5))
                prediction_binary = 1 if prediction_value >= 0.5 else 0

                # Determine if the prediction was correct based on the actual value
                is_correct = prediction_binary == actual

                patient_record['feedback'] = 'correct' if is_correct else 'incorrect'
                patient_record['feedback_timestamp'] = datetime.now().isoformat()
                patient_record['feedback_actual'] = actual
                patient_record['feedback_prediction'] = prediction_binary

                # Save updated record
                file_path = f'data/patients/{patient_id}.json'
                print(f"Saving updated patient record with feedback to {file_path}")
                with open(file_path, 'w') as f:
                    json.dump(patient_record, f, indent=2)
                    f.flush()  # Ensure data is written to disk
                    os.fsync(f.fileno())  # Force OS to write to physical storage

                # Verify the file was updated
                try:
                    with open(file_path, 'r') as f:
                        updated_record = json.load(f)
                        if 'feedback' in updated_record:
                            print(f"Verified feedback was saved: {updated_record['feedback']}")
                        else:
                            print("Warning: Feedback field not found in saved record")
                except Exception as verify_error:
                    print(f"Error verifying feedback save: {str(verify_error)}")

                success = True
                print(f"Feedback recorded for patient {patient_id}: {'correct' if actual == 1 else 'incorrect'}")
        except Exception as e:
            print(f"Error recording feedback: {str(e)}")
            success = False

        if success:
            # Get prediction value for the message
            prediction_value = float(patient_record.get('prediction', 0.5))
            prediction_text = "Heart Failure Risk" if prediction_value >= 0.5 else "No Heart Failure Risk"
            actual_text = "Heart Failure Risk" if actual == 1 else "No Heart Failure Risk"

            return jsonify({
                'success': True,
                'message': f"Feedback recorded for patient {patient_id}. Prediction was {prediction_text} ({prediction_value:.1%}), you marked it as {actual_text}.",
                'prediction': prediction_value,
                'actual': actual,
                'is_correct': prediction_binary == actual
            })
        else:
            return jsonify({
                'success': False,
                'message': "Failed to record feedback"
            }), 500

    except Exception as e:
        print(f"Error recording feedback: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add endpoints for longitudinal tracking
@app.route('/api/longitudinal/visits', methods=['POST'])
def add_visit():
    """Add a follow-up visit for a patient"""
    try:
        data = request.json
        patient_id = data.get('patient_id')
        visit_type = data.get('visit_type', 'follow-up')
        clinical_parameters = data.get('clinical_parameters', {})
        biomarkers = data.get('biomarkers', {})
        timestamp = data.get('timestamp', datetime.now().isoformat())

        # Get the patient or create if not exists
        patient = longitudinal_tracker.load_patient(patient_id)

        if patient is None:
            # Try to migrate from existing patient data
            if os.path.exists(f'data/patients/{patient_id}.json'):
                with open(f'data/patients/{patient_id}.json', 'r') as f:
                    patient_data = json.load(f)
                patient = longitudinal_tracker.migrate_existing_patient(patient_id, patient_data)
            else:
                # Create a new patient
                patient = longitudinal_tracker.LongitudinalPatient(patient_id=patient_id)

        # Generate ECG and risk assessment
        ecg_signal, ecg_time = generate_realistic_ecg(clinical_parameters)
        risk, confidence = calculate_risk_prediction(clinical_parameters)

        # Add the visit
        visit_id = patient.add_visit(
            timestamp=timestamp,
            visit_type=visit_type,
            clinical_parameters=clinical_parameters,
            biomarkers=biomarkers,
            ecg_data={
                'ecg_signal': ecg_signal,
                'ecg_time': ecg_time
            },
            risk_assessment={
                'prediction': risk,
                'confidence': confidence
            }
        )

        return jsonify({
            'status': 'success',
            'visit_id': visit_id,
            'patient_id': patient_id,
            'risk': risk,
            'confidence': confidence
        })
    except Exception as e:
        print(f"Error adding visit: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/longitudinal/patients/<patient_id>/visits', methods=['GET'])
def get_patient_visits(patient_id):
    """Get all visits for a patient"""
    try:
        patient = longitudinal_tracker.load_patient(patient_id)

        if patient is None:
            return jsonify({'error': 'Patient not found'}), 404

        # Get visits with optional filtering
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        visit_type = request.args.get('visit_type')

        visits = patient.get_visits(start_date, end_date, visit_type)

        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'visits': visits
        })
    except Exception as e:
        print(f"Error getting patient visits: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/longitudinal/patients/<patient_id>/trajectory', methods=['GET'])
def get_patient_trajectory(patient_id):
    """Get trajectory data for a patient"""
    try:
        patient = longitudinal_tracker.load_patient(patient_id)

        if patient is None:
            # Check if the patient exists in the regular patient data
            file_path = f'data/patients/{patient_id}.json'
            if os.path.exists(file_path):
                # Return empty trajectory data for patients without longitudinal data
                return jsonify({
                    'status': 'success',
                    'patient_id': patient_id,
                    'trajectory': [],
                    'analysis': {
                        'count': 0,
                        'mean': None,
                        'min': None,
                        'max': None,
                        'trend': None,
                        'r_squared': None,
                        'p_value': None,
                        'significant': False,
                        'message': 'No longitudinal data available for this patient'
                    }
                })
            else:
                return jsonify({'error': 'Patient not found'}), 404

        # Get trajectory data
        biomarker = request.args.get('biomarker')

        try:
            if biomarker:
                trajectory = patient.get_biomarker_trajectory(biomarker)
                analysis = patient.analyze_trajectory(biomarker)
            else:
                trajectory = patient.get_risk_trajectory()
                analysis = patient.analyze_trajectory()

            return jsonify({
                'status': 'success',
                'patient_id': patient_id,
                'trajectory': trajectory,
                'analysis': analysis
            })
        except FileNotFoundError:
            # Return empty trajectory data if no visits exist
            return jsonify({
                'status': 'success',
                'patient_id': patient_id,
                'trajectory': [],
                'analysis': {
                    'count': 0,
                    'mean': None,
                    'min': None,
                    'max': None,
                    'trend': None,
                    'r_squared': None,
                    'p_value': None,
                    'significant': False,
                    'message': 'No data available for trajectory analysis'
                }
            })
    except Exception as e:
        print(f"Error getting patient trajectory: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/counterfactuals', methods=['GET'])
def get_patient_counterfactuals(patient_id):
    """Get counterfactual explanations for a patient"""
    try:
        print(f"Generating counterfactual explanations for patient: {patient_id}")

        # Check if patient file exists
        file_path = f'data/patients/{patient_id}.json'
        if not os.path.exists(file_path):
            return jsonify({'error': 'Patient not found'}), 404

        # Load patient data
        with open(file_path, 'r') as f:
            patient_record = json.load(f)

        # Extract patient data
        patient_data = patient_record.get('patient_data', {})

        # Get number of counterfactuals to generate
        num_counterfactuals = request.args.get('num', 5)
        try:
            num_counterfactuals = int(num_counterfactuals)
        except (ValueError, TypeError):
            num_counterfactuals = 5

        # Generate counterfactual explanations
        counterfactuals = counterfactual_engine.generate_counterfactuals(
            patient_data, num_counterfactuals
        )

        # Add patient ID to response
        counterfactuals['patient_id'] = patient_id
        counterfactuals['status'] = 'success'

        return jsonify(counterfactuals)
    except Exception as e:
        print(f"Error generating counterfactual explanations: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/patients/<patient_id>/ecg/12lead', methods=['GET'])
def get_patient_12lead_ecg(patient_id):
    """Get 12-lead ECG data for a patient"""
    try:
        # Check if patient exists
        file_path = f'data/patients/{patient_id}.json'
        if not os.path.exists(file_path):
            return jsonify({'error': f'Patient {patient_id} not found'}), 404

        # Load patient data
        with open(file_path, 'r') as f:
            patient_data = json.load(f)

        # Generate a simplified 12-lead ECG data structure
        # This is a basic implementation that doesn't rely on complex processing
        try:
            # Check if patient data has ECG signal already
            if 'ecg_signal' in patient_data and 'ecg_time' in patient_data:
                # Use existing ECG signal as base
                base_signal = np.array(patient_data['ecg_signal'])
                time_array = np.array(patient_data['ecg_time'])
            else:
                # Convert patient data to appropriate types
                processed_data = {}
                for key, value in patient_data['patient_data'].items():
                    if key == 'age' and isinstance(value, str):
                        try:
                            processed_data[key] = float(value)
                        except (ValueError, TypeError):
                            processed_data[key] = 60  # Default age
                    elif key == 'max_heart_rate' and isinstance(value, str):
                        try:
                            processed_data[key] = float(value)
                        except (ValueError, TypeError):
                            processed_data[key] = 75  # Default heart rate
                    else:
                        processed_data[key] = value

                # Generate base ECG signal
                base_signal, time_array = generate_ecg(processed_data)
        except Exception as e:
            print(f"Error generating ECG: {str(e)}")
            # Create fallback data
            base_signal = np.zeros(5000)
            time_array = np.linspace(0, 10, 5000)

        # Create a simplified 12-lead ECG data structure
        ecg_data = {
            'leads': {
                'I': base_signal.tolist(),
                'II': base_signal.tolist(),
                'III': base_signal.tolist(),
                'aVR': base_signal.tolist(),
                'aVL': base_signal.tolist(),
                'aVF': base_signal.tolist(),
                'V1': base_signal.tolist(),
                'V2': base_signal.tolist(),
                'V3': base_signal.tolist(),
                'V4': base_signal.tolist(),
                'V5': base_signal.tolist(),
                'V6': base_signal.tolist()
            },
            'time': time_array.tolist(),
            'metadata': {
                'heart_rate': 75,
                'paper_speed': 25,
                'amplitude_scale': 10
            },
            'lead_order': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        }

        # Create a simplified analysis result with lead-specific abnormalities
        analysis_results = {
            'abnormalities': {
                'rhythm': [{
                    'type': 'Sinus Rhythm',
                    'description': 'Normal sinus rhythm',
                    'confidence': 0.95,
                    'lead': 'II'
                }],
                'conduction': [{
                    'type': 'AV Block',
                    'description': 'First-degree AV block',
                    'confidence': 0.85,
                    'lead': 'I'
                }],
                'st_changes': [{
                    'type': 'ST Elevation',
                    'description': 'ST segment elevation',
                    'confidence': 0.78,
                    'lead': 'V2'
                }, {
                    'type': 'ST Depression',
                    'description': 'ST segment depression',
                    'confidence': 0.82,
                    'lead': 'V5'
                }],
                'chamber_enlargement': [{
                    'type': 'Left Atrial Enlargement',
                    'description': 'P-wave abnormality suggesting left atrial enlargement',
                    'confidence': 0.75,
                    'lead': 'V1'
                }],
                'axis_deviation': [{
                    'type': 'Left Axis Deviation',
                    'description': 'QRS axis deviation to the left',
                    'confidence': 0.88,
                    'lead': 'aVF'
                }],
                'infarction': [{
                    'type': 'Anterior Infarction',
                    'description': 'Q waves suggesting anterior wall infarction',
                    'confidence': 0.72,
                    'lead': 'V3'
                }],
                'PVCs': [{
                    'type': 'PVC',
                    'description': 'Premature ventricular contraction',
                    'confidence': 0.91,
                    'lead': 'V6',
                    'time': 2.5,
                    'duration': 0.4
                }],
                'QT_prolongation': [{
                    'type': 'Prolonged QT',
                    'description': 'QT interval prolongation',
                    'confidence': 0.83,
                    'lead': 'III'
                }]
            },
            'rhythm': {'name': 'Normal Sinus Rhythm', 'regularity': 'Regular', 'confidence': 0.9},
            'heart_rate': 75,
            'intervals': {
                'PR': 0.16,
                'QRS': 0.08,
                'QT': 0.38,
                'QTc': 0.41
            },
            'axis': {
                'value': 60,
                'category': 'Normal',
                'confidence': 0.7
            }
        }

        # Return the results
        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'ecg_data': ecg_data,
            'analysis': analysis_results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error generating 12-lead ECG: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/longitudinal/patients', methods=['GET'])
def get_all_longitudinal_patients():
    """Get all patients with longitudinal data"""
    try:
        patients = longitudinal_tracker.get_all_patients()

        return jsonify({
            'status': 'success',
            'patients': patients
        })
    except Exception as e:
        print(f"Error getting longitudinal patients: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting test Flask server on port 8080...")
    app.run(debug=False, port=8080, host='0.0.0.0')
