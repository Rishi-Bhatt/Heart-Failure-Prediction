import os
import json
import random
from datetime import datetime

print("Simple test of Heart Failure Prediction System")
print("=============================================")

# Create necessary directories
os.makedirs('backend/data/patients', exist_ok=True)
print("Directories created successfully!")

# Create a simple patient record
patient_data = {
    'patient_id': f"patient_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    'name': 'Test Patient',
    'age': 65,
    'gender': 'Male',
    'prediction': random.random(),
    'confidence': 0.8
}

# Save to file
file_path = f"backend/data/patients/{patient_data['patient_id']}.json"
with open(file_path, 'w') as f:
    json.dump(patient_data, f, indent=2)

print(f"Patient data saved to {file_path}")

# Read the file back
with open(file_path, 'r') as f:
    loaded_data = json.load(f)

print(f"Successfully loaded patient data: {loaded_data['name']}")
print("Test completed successfully!")
