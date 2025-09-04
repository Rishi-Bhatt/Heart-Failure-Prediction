"""
Update Ensemble Weights

This script updates the ensemble weights in the database to focus more on
the Random Forest model.
"""

import os
import sys
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def update_ensemble_weights():
    """
    Update the ensemble weights in the database
    """
    print("\nUpdating Ensemble Weights")
    print("=" * 60)
    
    # Define new weights
    new_weights = {
        'rule_based': 0.15,
        'logistic_regression': 0.15,
        'random_forest': 0.70
    }
    
    # Define paths
    weights_path = 'data/ensemble_weights.json'
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save weights to file
    try:
        with open(weights_path, 'w') as f:
            json.dump(new_weights, f)
        print(f"Saved new weights to {weights_path}")
    except Exception as e:
        print(f"Error saving weights: {str(e)}")
        return False
    
    # Print new weights
    print("\nNew Ensemble Weights:")
    for component, weight in new_weights.items():
        print(f"  {component}: {weight:.2f}")
    
    print("\nEnsemble weights updated successfully!")
    return True

if __name__ == "__main__":
    update_ensemble_weights()
