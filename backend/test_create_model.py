import os
import sys
import traceback

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting model creation test...")

try:
    print("Creating models directory if it doesn't exist...")
    os.makedirs('models', exist_ok=True)
    
    print("Importing HeartFailureModel...")
    from models.heart_failure_model import HeartFailureModel
    
    print("Initializing model (this will train a new model)...")
    model = HeartFailureModel()
    
    print("Model initialization complete!")
    
    # Check if model files were created
    if os.path.exists('models/heart_failure_model.joblib') and os.path.exists('models/heart_failure_scaler.joblib'):
        print("Model files created successfully!")
    else:
        print("Model files were not created.")
    
except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()

print("Test completed.")
