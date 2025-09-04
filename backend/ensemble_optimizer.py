"""
Ensemble Optimizer for Heart Failure Prediction System

This module optimizes the weights of the ensemble model components
based on performance metrics to maximize overall accuracy.
"""

import os
import json
import numpy as np
from datetime import datetime

# Default weights
DEFAULT_WEIGHTS = {
    'rule_based': 0.15,
    'logistic_regression': 0.15,
    'random_forest': 0.70
}

# Alternative weights with gradient boosting
DEFAULT_WEIGHTS_WITH_GB = {
    'rule_based': 0.15,
    'logistic_regression': 0.10,
    'random_forest': 0.65,
    'gradient_boosting': 0.10
}

# File paths
WEIGHTS_FILE = 'data/ensemble_weights.json'
OPTIMIZATION_HISTORY_FILE = 'data/optimization_history.json'
HYBRID_CONFIG_FILE = 'models/hybrid_config.json'

def load_weights(sync=False):
    """
    Load ensemble weights from file

    Parameters:
    -----------
    sync : bool
        Whether to synchronize weights with hybrid model

    Returns:
    --------
    weights : dict
        Ensemble weights
    """
    # Try to load weights from file
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE, 'r') as f:
                weights = json.load(f)

            # Check if weights need to be synchronized
            if sync:
                return synchronize_weights()
            return weights
        except Exception as e:
            print(f"Error loading weights: {str(e)}")
            return DEFAULT_WEIGHTS
    else:
        # If weights file doesn't exist, check if we should synchronize
        if sync and os.path.exists(HYBRID_CONFIG_FILE):
            return synchronize_weights()
        return DEFAULT_WEIGHTS

def save_weights(weights, update_hybrid=True):
    """
    Save ensemble weights to file

    Parameters:
    -----------
    weights : dict
        Ensemble weights to save
    update_hybrid : bool
        Whether to update hybrid config file
    """
    os.makedirs(os.path.dirname(WEIGHTS_FILE), exist_ok=True)
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weights, f, indent=2)

    # Update hybrid config if requested
    if update_hybrid:
        update_hybrid_config(weights)

def optimize_weights(performance_metrics):
    """
    Optimize ensemble weights based on performance metrics

    Parameters:
    -----------
    performance_metrics : dict
        Dictionary with performance metrics for each component

    Returns:
    --------
    optimized_weights : dict
        Optimized ensemble weights
    """
    # Load current weights
    current_weights = load_weights()

    # If no performance metrics, return current weights
    if not performance_metrics:
        return current_weights

    # Extract accuracy metrics
    accuracies = {
        component: metrics.get('accuracy', 0.5)
        for component, metrics in performance_metrics.items()
    }

    # Calculate total accuracy
    total_accuracy = sum(accuracies.values())

    # If total accuracy is 0, use default weights
    if total_accuracy == 0:
        return DEFAULT_WEIGHTS

    # Calculate new weights based on relative accuracy
    new_weights = {
        component: max(0.1, min(0.7, accuracy / total_accuracy))
        for component, accuracy in accuracies.items()
    }

    # Normalize weights to sum to 1
    weight_sum = sum(new_weights.values())
    normalized_weights = {
        component: weight / weight_sum
        for component, weight in new_weights.items()
    }

    # Apply smoothing to avoid drastic changes
    smoothing_factor = 0.7  # 70% old weights, 30% new weights
    optimized_weights = {
        component: smoothing_factor * current_weights.get(component, DEFAULT_WEIGHTS.get(component, 0.33)) +
                  (1 - smoothing_factor) * normalized_weights.get(component, 0.33)
        for component in current_weights.keys()
    }

    # Normalize again to ensure sum is 1
    weight_sum = sum(optimized_weights.values())
    optimized_weights = {
        component: weight / weight_sum
        for component, weight in optimized_weights.items()
    }

    # Save optimization history
    save_optimization_history(current_weights, optimized_weights, performance_metrics)

    # Save new weights
    save_weights(optimized_weights)

    return optimized_weights

def save_optimization_history(old_weights, new_weights, performance_metrics):
    """
    Save optimization history to file
    """
    # Load existing history
    if os.path.exists(OPTIMIZATION_HISTORY_FILE):
        try:
            with open(OPTIMIZATION_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []

    # Add new entry
    history.append({
        'timestamp': datetime.now().isoformat(),
        'old_weights': old_weights,
        'new_weights': new_weights,
        'performance_metrics': performance_metrics
    })

    # Save history
    os.makedirs(os.path.dirname(OPTIMIZATION_HISTORY_FILE), exist_ok=True)
    with open(OPTIMIZATION_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def get_current_weights(sync=True):
    """
    Get current ensemble weights

    Parameters:
    -----------
    sync : bool
        Whether to synchronize weights with hybrid model

    Returns:
    --------
    weights : dict
        Ensemble weights
    """
    return load_weights(sync=sync)

def reset_weights():
    """
    Reset weights to default values
    """
    save_weights(DEFAULT_WEIGHTS)
    return DEFAULT_WEIGHTS

def synchronize_weights():
    """
    Synchronize weights across different components
    """
    # Load current weights
    current_weights = load_weights()

    # Check if hybrid config file exists
    if os.path.exists(HYBRID_CONFIG_FILE):
        try:
            with open(HYBRID_CONFIG_FILE, 'r') as f:
                hybrid_config = json.load(f)
                hybrid_weights = hybrid_config.get('ensemble_weights', {})

                # Convert hybrid weights format to ensemble weights format
                if hybrid_weights:
                    ensemble_weights = {}

                    # Map hybrid model weights to ensemble weights
                    if 'rule_based' in hybrid_weights:
                        ensemble_weights['rule_based'] = hybrid_weights['rule_based']

                    if 'ml_model' in hybrid_weights:
                        ensemble_weights['logistic_regression'] = hybrid_weights['ml_model']

                    if 'random_forest' in hybrid_weights:
                        ensemble_weights['random_forest'] = hybrid_weights['random_forest']

                    if 'gradient_boosting' in hybrid_weights:
                        ensemble_weights['gradient_boosting'] = hybrid_weights['gradient_boosting']

                    # Normalize weights
                    weight_sum = sum(ensemble_weights.values())
                    if weight_sum > 0:
                        for key in ensemble_weights:
                            ensemble_weights[key] /= weight_sum

                    # Save synchronized weights
                    save_weights(ensemble_weights)
                    print(f"Synchronized weights from hybrid config: {ensemble_weights}")
                    return ensemble_weights
        except Exception as e:
            print(f"Error synchronizing weights from hybrid config: {str(e)}")

    return current_weights

def get_hybrid_weights():
    """
    Get weights in hybrid model format
    """
    # Load current weights
    current_weights = load_weights()

    # Convert to hybrid model format
    hybrid_weights = {
        'rule_based': current_weights.get('rule_based', 0.40),
        'ml_model': current_weights.get('logistic_regression', 0.30),
        'random_forest': current_weights.get('random_forest', 0.30)
    }

    # Add gradient boosting if it exists in ensemble weights
    if 'gradient_boosting' in current_weights:
        hybrid_weights['gradient_boosting'] = current_weights.get('gradient_boosting', 0.0)

    # Normalize weights
    weight_sum = sum(hybrid_weights.values())
    if weight_sum > 0:
        for key in hybrid_weights:
            hybrid_weights[key] /= weight_sum

    return hybrid_weights

def update_hybrid_config(weights):
    """
    Update hybrid config file with ensemble weights

    Parameters:
    -----------
    weights : dict
        Ensemble weights to convert and save
    """
    # Check if hybrid config file exists
    if not os.path.exists(HYBRID_CONFIG_FILE):
        # If not, create directory and empty config
        os.makedirs(os.path.dirname(HYBRID_CONFIG_FILE), exist_ok=True)
        hybrid_config = {}
    else:
        # Load existing config
        try:
            with open(HYBRID_CONFIG_FILE, 'r') as f:
                hybrid_config = json.load(f)
        except Exception as e:
            print(f"Error loading hybrid config: {str(e)}")
            hybrid_config = {}

    # Convert ensemble weights to hybrid format
    hybrid_weights = {
        'rule_based': weights.get('rule_based', 0.40),
        'ml_model': weights.get('logistic_regression', 0.30),
        'random_forest': weights.get('random_forest', 0.30)
    }

    # Add gradient boosting if it exists in ensemble weights
    if 'gradient_boosting' in weights:
        hybrid_weights['gradient_boosting'] = weights.get('gradient_boosting', 0.0)

    # Normalize weights
    weight_sum = sum(hybrid_weights.values())
    if weight_sum > 0:
        for key in hybrid_weights:
            hybrid_weights[key] /= weight_sum

    # Update hybrid config
    hybrid_config['ensemble_weights'] = hybrid_weights
    hybrid_config['timestamp'] = datetime.now().isoformat()

    # Save hybrid config
    try:
        with open(HYBRID_CONFIG_FILE, 'w') as f:
            json.dump(hybrid_config, f, indent=2)
        print(f"Updated hybrid config with weights: {hybrid_weights}")
    except Exception as e:
        print(f"Error updating hybrid config: {str(e)}")
