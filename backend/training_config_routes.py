"""
Training Configuration Routes for Heart Failure Prediction System

This module provides API endpoints to get and update training configuration.
"""

from flask import jsonify, request
from training_config import get_training_config, update_training_config

def register_training_config_routes(app):
    """
    Register training configuration routes with Flask app

    Parameters:
    -----------
    app : Flask
        Flask application
    """

    @app.route('/api/training/config', methods=['GET'])
    def get_config():
        """
        Get current training configuration
        """
        try:
            config = get_training_config()
            return jsonify(config)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/training/config', methods=['POST'])
    def update_config():
        """
        Update training configuration
        """
        try:
            # Get updates from request
            updates = request.json or {}

            # Update configuration
            config = update_training_config(updates)

            return jsonify({
                'success': True,
                'message': 'Training configuration updated successfully',
                'config': config
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error updating training configuration: {str(e)}'
            }), 500
