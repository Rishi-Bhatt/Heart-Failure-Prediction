"""
Training Configuration Manager for Heart Failure Prediction System

This module provides functions to read and update training configuration settings.
"""

import os
import json

# Default configuration
DEFAULT_CONFIG = {
    "epochs": 50,
    "use_neural_network": False,
    "batch_size": 32,
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "learning_rate": 0.001,
    "optimizer": "adam"
}

# Path to configuration file
CONFIG_PATH = os.path.join('data', 'training_config.json')

def get_training_config():
    """
    Get training configuration from file
    
    Returns:
    --------
    config : dict
        Training configuration
    """
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            return config
        else:
            # Create default config if file doesn't exist
            save_training_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
    except Exception as e:
        print(f"Error reading training config: {str(e)}")
        return DEFAULT_CONFIG

def save_training_config(config):
    """
    Save training configuration to file
    
    Parameters:
    -----------
    config : dict
        Training configuration
    
    Returns:
    --------
    success : bool
        Whether the save was successful
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        
        # Save config to file
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving training config: {str(e)}")
        return False

def update_training_config(updates):
    """
    Update training configuration with new values
    
    Parameters:
    -----------
    updates : dict
        Dictionary of configuration updates
    
    Returns:
    --------
    config : dict
        Updated configuration
    """
    try:
        # Get current config
        config = get_training_config()
        
        # Update config with new values
        for key, value in updates.items():
            if key in config:
                # Convert epochs to int if provided
                if key == 'epochs' and value is not None:
                    try:
                        value = int(value)
                        if value <= 0:
                            value = DEFAULT_CONFIG['epochs']
                    except (ValueError, TypeError):
                        value = DEFAULT_CONFIG['epochs']
                
                # Convert use_neural_network to bool if provided
                if key == 'use_neural_network':
                    value = bool(value)
                
                config[key] = value
        
        # Save updated config
        save_training_config(config)
        return config
    except Exception as e:
        print(f"Error updating training config: {str(e)}")
        return get_training_config()
