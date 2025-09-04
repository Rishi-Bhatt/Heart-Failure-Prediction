"""
Synchronize Ensemble Weights

This script synchronizes the weights between the hybrid model and the ensemble model.
It ensures that both models use consistent weights for prediction.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ensemble optimizer
try:
    from ensemble_optimizer import synchronize_weights, get_hybrid_weights, update_hybrid_config
    ENSEMBLE_OPTIMIZER_AVAILABLE = True
except ImportError:
    ENSEMBLE_OPTIMIZER_AVAILABLE = False
    print("Error: ensemble_optimizer module not found.")
    sys.exit(1)

def main():
    """
    Main function to synchronize weights
    """
    print("Synchronizing ensemble weights...")
    
    # Synchronize weights from hybrid config to ensemble weights
    ensemble_weights = synchronize_weights()
    print(f"Ensemble weights: {ensemble_weights}")
    
    # Get hybrid weights
    hybrid_weights = get_hybrid_weights()
    print(f"Hybrid weights: {hybrid_weights}")
    
    # Update hybrid config with ensemble weights
    update_hybrid_config(ensemble_weights)
    print("Weights synchronized successfully.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
