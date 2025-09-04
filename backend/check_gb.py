"""
Simple script to check if gradient boosting is being used in the risk assessment
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ensemble optimizer
from ensemble_optimizer import get_current_weights

# Print current weights
print("Current ensemble weights:")
current_weights = get_current_weights()
print(current_weights)

# Check if gradient boosting is in the weights
gb_in_weights = 'gradient_boosting' in current_weights
print(f"Gradient Boosting in weights: {gb_in_weights}")

# If gradient boosting is in weights, print its weight
if gb_in_weights:
    print(f"Gradient Boosting weight: {current_weights['gradient_boosting']}")
