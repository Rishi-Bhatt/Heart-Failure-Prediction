"""
Gradient Boosting Routes for Heart Failure Prediction System

This module provides API routes for training and managing the gradient boosting model.
"""

from flask import jsonify, request
import os
import json
from datetime import datetime

# Import gradient boosting model
try:
    from models.gradient_boosting_model import train_gradient_boosting_model, gb_model
    GRADIENT_BOOSTING_AVAILABLE = True
except ImportError:
    print("Warning: Gradient Boosting model not found. Routes will return appropriate errors.")
    GRADIENT_BOOSTING_AVAILABLE = False

def register_gradient_boosting_routes(app):
    """
    Register gradient boosting routes with the Flask app
    
    Parameters:
    -----------
    app : Flask
        Flask application instance
    """
    
    @app.route('/api/gradient-boosting/train', methods=['POST'])
    def train_gradient_boosting():
        """
        Endpoint to train the gradient boosting model
        """
        if not GRADIENT_BOOSTING_AVAILABLE:
            return jsonify({
                'success': False,
                'message': 'Gradient Boosting model is not available. Please check server logs for details.'
            }), 500
        
        try:
            # Get configuration from request if provided
            config = request.json or {}
            
            # Load patient data
            patient_files = [f for f in os.listdir('data/patients') if f.endswith('.json')]
            
            if len(patient_files) < 10:
                return jsonify({
                    'success': False,
                    'message': f'Insufficient data for training Gradient Boosting model. Need at least 10 records, but only found {len(patient_files)}.',
                    'num_records': len(patient_files)
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
            
            print(f"Loaded {len(patient_data_list)} patient records for Gradient Boosting model training")
            
            # Train the model
            training_result = train_gradient_boosting_model(patient_data_list)
            
            # Add timestamp to result
            training_result['timestamp'] = datetime.now().isoformat()
            
            # Save training result to history
            save_training_result(training_result)
            
            return jsonify(training_result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Error training Gradient Boosting model: {str(e)}'
            }), 500
    
    @app.route('/api/gradient-boosting/status', methods=['GET'])
    def get_gradient_boosting_status():
        """
        Endpoint to get the status of the gradient boosting model
        """
        if not GRADIENT_BOOSTING_AVAILABLE:
            return jsonify({
                'available': False,
                'message': 'Gradient Boosting model is not available'
            })
        
        try:
            # Get model status
            status = {
                'available': True,
                'trained': gb_model.is_trained,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add training metrics if available
            if gb_model.is_trained and gb_model.training_metrics:
                status['metrics'] = gb_model.training_metrics
                
            # Add feature importance if available
            if gb_model.is_trained:
                status['feature_importance'] = gb_model.get_feature_importance()
            
            return jsonify(status)
            
        except Exception as e:
            return jsonify({
                'available': True,
                'trained': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    @app.route('/api/gradient-boosting/history', methods=['GET'])
    def get_gradient_boosting_history():
        """
        Endpoint to get the training history of the gradient boosting model
        """
        try:
            history_path = 'data/gb_training_history.json'
            
            if not os.path.exists(history_path):
                return jsonify([])
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            return jsonify(history)
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    print("Gradient Boosting routes registered successfully")

def save_training_result(training_result):
    """
    Save training result to history file
    
    Parameters:
    -----------
    training_result : dict
        Training result dictionary
    """
    try:
        history_path = 'data/gb_training_history.json'
        
        # Initialize or load history
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new entry
        history.append(training_result)
        
        # Save history
        os.makedirs('data', exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Error saving Gradient Boosting training result: {str(e)}")
