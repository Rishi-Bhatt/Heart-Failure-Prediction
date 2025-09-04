"""
Set Gradient Boosting Weights

This script manually sets the ensemble weights to include gradient boosting.
"""

import os
import json
from datetime import datetime

# Define weights with gradient boosting
weights_with_gb = {
    'rule_based': 0.3,
    'logistic_regression': 0.2,
    'random_forest': 0.25,
    'gradient_boosting': 0.25
}

# Define hybrid weights with gradient boosting
hybrid_weights_with_gb = {
    'rule_based': 0.3,
    'ml_model': 0.2,
    'random_forest': 0.25,
    'gradient_boosting': 0.25
}

# Save ensemble weights
ensemble_weights_file = 'data/ensemble_weights.json'
os.makedirs(os.path.dirname(ensemble_weights_file), exist_ok=True)
with open(ensemble_weights_file, 'w') as f:
    json.dump(weights_with_gb, f, indent=2)
print(f"Saved ensemble weights to {ensemble_weights_file}")

# Save hybrid config
hybrid_config_file = 'models/hybrid_config.json'
os.makedirs(os.path.dirname(hybrid_config_file), exist_ok=True)
hybrid_config = {
    'ensemble_weights': hybrid_weights_with_gb,
    'timestamp': datetime.now().isoformat()
}
with open(hybrid_config_file, 'w') as f:
    json.dump(hybrid_config, f, indent=2)
print(f"Saved hybrid config to {hybrid_config_file}")

# Verify the weights
print("\nVerifying weights:")
try:
    with open(ensemble_weights_file, 'r') as f:
        ensemble_weights = json.load(f)
    print(f"Ensemble weights: {ensemble_weights}")
    print(f"Gradient Boosting in ensemble weights: {'gradient_boosting' in ensemble_weights}")
except Exception as e:
    print(f"Error reading ensemble weights: {str(e)}")

try:
    with open(hybrid_config_file, 'r') as f:
        hybrid_config = json.load(f)
    print(f"Hybrid config weights: {hybrid_config['ensemble_weights']}")
    print(f"Gradient Boosting in hybrid config: {'gradient_boosting' in hybrid_config['ensemble_weights']}")
except Exception as e:
    print(f"Error reading hybrid config: {str(e)}")

print("\nDone!")
