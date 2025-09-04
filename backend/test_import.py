import os
import sys
import traceback

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting import test...")

try:
    print("Importing HeartFailureModel...")
    from models.heart_failure_model import HeartFailureModel
    print("Import successful!")
except Exception as e:
    print(f"Error importing HeartFailureModel: {str(e)}")
    traceback.print_exc()

print("Test completed.")
