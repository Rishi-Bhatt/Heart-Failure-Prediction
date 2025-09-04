"""
Counterfactual explanation routes for the heart failure prediction system.

This module provides endpoints for generating and retrieving counterfactual
explanations for patient risk predictions.
"""

import os
import json
import traceback
from flask import jsonify
from counterfactual_engine import CounterfactualEngine

# Initialize the counterfactual engine
counterfactual_engine = None

def register_counterfactual_routes(app):
    """Register counterfactual explanation routes with the Flask app."""

    # Initialize the counterfactual engine
    global counterfactual_engine
    counterfactual_engine = CounterfactualEngine()

    @app.route('/api/patients/<patient_id>/counterfactuals', methods=['GET'])
    def get_counterfactuals(patient_id):
        """
        Endpoint to get counterfactual explanations for a specific patient.

        This endpoint generates "what-if" scenarios showing how changes to
        specific risk factors would affect the predicted heart failure risk.
        """
        try:
            # Check if patient file exists
            file_path = f'data/patients/{patient_id}.json'
            if not os.path.exists(file_path):
                return jsonify({
                    'status': 'error',
                    'message': f'Patient {patient_id} not found'
                }), 404

            # Load patient data
            with open(file_path, 'r') as f:
                patient_record = json.load(f)

            # Extract patient data
            patient_data = patient_record.get('patient_data', {})

            # Add biomarkers if available
            if 'biomarkers' in patient_record:
                patient_data['biomarkers'] = patient_record['biomarkers']

            # Add ECG abnormalities if available
            if 'ecg_abnormalities' in patient_record:
                patient_data['ecg_abnormalities'] = patient_record['ecg_abnormalities']

            # Generate counterfactuals
            print(f"Generating counterfactuals for patient {patient_id}...")
            counterfactuals = counterfactual_engine.generate_counterfactuals(patient_data)

            # Add status field
            counterfactuals['status'] = 'success'

            # Add patient ID for reference
            counterfactuals['patient_id'] = patient_id

            return jsonify(counterfactuals)

        except Exception as e:
            print(f"Error generating counterfactuals: {str(e)}")
            traceback.print_exc()

            # Return error response
            return jsonify({
                'status': 'error',
                'message': f'Error generating counterfactuals: {str(e)}'
            }), 500
