"""
API Routes for Temporal Forecasting

This module provides API endpoints for temporal forecasting functionality.
"""

from flask import Blueprint, request, jsonify
import json
import os
from datetime import datetime

# Import forecasting functionality
from temporal_forecaster import (
    generate_forecast,
    generate_scenario_forecast,
    get_forecast,
    get_all_forecasts,
    train_forecasting_model
)

# Import longitudinal patient functionality
from longitudinal_tracker import load_patient, get_all_patients, migrate_existing_patient

# Import patient data functionality
import os
import json
from config import DATA_PATHS

# Create blueprint
forecast_bp = Blueprint('forecast', __name__)

@forecast_bp.route('/api/forecast/train', methods=['POST'])
def train_forecast_model():
    """Train the forecasting model on all available patient data."""
    try:
        # Get all patients
        patients = get_all_patients()

        if not patients:
            return jsonify({
                'status': 'error',
                'message': 'No patient data available for training.'
            })

        # Load visit data for each patient
        patient_histories = []

        for patient_metadata in patients:
            patient_id = patient_metadata.get('patient_id')
            if patient_id:
                patient = load_patient(patient_id)
                if patient:
                    visits = patient.get_visits()
                    if visits:
                        patient_histories.append(visits)

        # Get forecast horizon from request
        data = request.get_json() or {}
        forecast_horizon = data.get('forecast_horizon', 6)

        # Train the model
        result = train_forecasting_model(patient_histories, forecast_horizon)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error training forecasting model: {str(e)}'
        })

@forecast_bp.route('/api/patients/<patient_id>/forecast', methods=['GET'])
def get_patient_forecast(patient_id):
    """Generate a forecast for a specific patient."""
    try:
        # Get forecast horizon from query parameters
        horizon = request.args.get('horizon', default=6, type=int)

        # Load patient data from longitudinal system
        patient = load_patient(patient_id)

        # If patient not found in longitudinal system, try to migrate from main system
        if not patient:
            print(f"Patient {patient_id} not found in longitudinal system, attempting to migrate")

            # Try to load from main patient data
            patient_file_path = os.path.join(DATA_PATHS['patient_data_directory'], f"{patient_id}.json")

            if os.path.exists(patient_file_path):
                try:
                    with open(patient_file_path, 'r') as f:
                        patient_data = json.load(f)

                    # Migrate patient to longitudinal system
                    patient = migrate_existing_patient(patient_id, patient_data)
                    print(f"Successfully migrated patient {patient_id} to longitudinal system")
                except Exception as migration_error:
                    print(f"Error migrating patient {patient_id}: {str(migration_error)}")

        # If still no patient found, return error
        if not patient:
            return jsonify({
                'status': 'error',
                'message': f'Patient {patient_id} not found. Please ensure the patient has sufficient visit history for forecasting.'
            })

        # Get patient visits
        visits = patient.get_visits()

        if not visits or len(visits) < 2:  # Need at least 2 visits for meaningful forecasting
            return jsonify({
                'status': 'error',
                'message': f'Insufficient visit data available for patient {patient_id}. At least 2 visits are required for forecasting.'
            })

        # Generate forecast
        result = generate_forecast(visits, horizon)

        # Add patient information
        if result['status'] == 'success':
            result['patient_id'] = patient_id
            result['patient_name'] = patient.demographic_data.get('name', 'Unknown')

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating forecast: {str(e)}'
        })

@forecast_bp.route('/api/patients/<patient_id>/forecast/scenario', methods=['POST'])
def get_patient_scenario_forecast(patient_id):
    """Generate a scenario-based forecast for a specific patient."""
    try:
        # Get scenario data from request
        data = request.get_json()

        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No scenario data provided.'
            })

        # Load patient data from longitudinal system
        patient = load_patient(patient_id)

        # If patient not found in longitudinal system, try to migrate from main system
        if not patient:
            print(f"Patient {patient_id} not found in longitudinal system, attempting to migrate for scenario")

            # Try to load from main patient data
            patient_file_path = os.path.join(DATA_PATHS['patient_data_directory'], f"{patient_id}.json")

            if os.path.exists(patient_file_path):
                try:
                    with open(patient_file_path, 'r') as f:
                        patient_data = json.load(f)

                    # Migrate patient to longitudinal system
                    patient = migrate_existing_patient(patient_id, patient_data)
                    print(f"Successfully migrated patient {patient_id} to longitudinal system for scenario")
                except Exception as migration_error:
                    print(f"Error migrating patient {patient_id} for scenario: {str(migration_error)}")

        # If still no patient found, return error
        if not patient:
            return jsonify({
                'status': 'error',
                'message': f'Patient {patient_id} not found. Please ensure the patient has sufficient visit history for scenario forecasting.'
            })

        # Get patient visits
        visits = patient.get_visits()

        if not visits or len(visits) < 2:  # Need at least 2 visits for meaningful forecasting
            return jsonify({
                'status': 'error',
                'message': f'Insufficient visit data available for patient {patient_id}. At least 2 visits are required for scenario forecasting.'
            })

        # Get forecast horizon from query parameters
        horizon = request.args.get('horizon', default=6, type=int)

        # Generate scenario forecast with the specified horizon
        result = generate_scenario_forecast(visits, data, horizon)

        # Add patient information
        if result['status'] == 'success':
            result['patient_id'] = patient_id
            result['patient_name'] = patient.demographic_data.get('name', 'Unknown')

            # Add scenario name to result
            if 'name' in data:
                result['scenario_name'] = data['name']

            # Add interventions to result
            if 'interventions' in data:
                result['interventions'] = data['interventions']

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error generating scenario forecast: {str(e)}'
        })

@forecast_bp.route('/api/forecasts/<forecast_id>', methods=['GET'])
def get_forecast_by_id(forecast_id):
    """Retrieve a specific forecast by ID."""
    try:
        forecast_data = get_forecast(forecast_id)

        if not forecast_data:
            return jsonify({
                'status': 'error',
                'message': f'Forecast {forecast_id} not found.'
            })

        return jsonify({
            'status': 'success',
            'forecast': forecast_data
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving forecast: {str(e)}'
        })

@forecast_bp.route('/api/forecasts', methods=['GET'])
def get_all_forecasts_route():
    """Retrieve all forecasts, optionally filtered by patient ID."""
    try:
        # Get patient ID from query parameters
        patient_id = request.args.get('patient_id', default=None)

        # Get all forecasts
        forecasts = get_all_forecasts(patient_id)

        return jsonify({
            'status': 'success',
            'forecasts': forecasts
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving forecasts: {str(e)}'
        })

def register_forecast_routes(app):
    """Register forecast routes with the Flask app."""
    app.register_blueprint(forecast_bp)
