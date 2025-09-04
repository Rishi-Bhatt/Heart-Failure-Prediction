import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in current directory:", os.listdir())

try:
    print("\nTrying to import from utils...")
    from utils.ecg_generator import generate_ecg, analyze_ecg
    print("Successfully imported from utils")
except Exception as e:
    print("Error importing from utils:", str(e))

try:
    print("\nTrying to import from models...")
    from models.heart_failure_model import HeartFailureModel
    print("Successfully imported from models")
except Exception as e:
    print("Error importing from models:", str(e))

try:
    print("\nTrying to import from retraining...")
    from retraining.model_retrainer import ModelRetrainer
    print("Successfully imported from retraining")
except Exception as e:
    print("Error importing from retraining:", str(e))

print("\nTest complete")
