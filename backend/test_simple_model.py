import os
import sys
import traceback
import numpy as np

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting simple model test...")

try:
    print("Importing SimpleHeartFailureModel...")
    from models.simple_model import SimpleHeartFailureModel

    print("Initializing simple model...")
    model = SimpleHeartFailureModel()

    print("Making prediction...")
    features = np.array([[65, 1, 220, 140]])  # Dummy features
    prediction, confidence, shap_values = model.predict(features)

    print(f"Prediction: {prediction:.4f}")
    print(f"Confidence: {confidence:.4f}")

    print("Test completed successfully!")

except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()

print("Test completed.")
