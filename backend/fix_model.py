"""
Script to fix the logistic regression model by adding the classes_ attribute.
"""
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

# Define paths for model data
MODEL_DIR = 'models'
ML_MODEL_FILE = os.path.join(MODEL_DIR, 'clinical_lr_model.json')

print(f"Checking if model file exists at {ML_MODEL_FILE}...")
if not os.path.exists(ML_MODEL_FILE):
    print(f"Model file {ML_MODEL_FILE} not found")
    exit(1)

print("Loading model data...")
try:
    with open(ML_MODEL_FILE, 'r') as f:
        model_data = json.load(f)
    print("Model data loaded successfully")
except Exception as e:
    print(f"Error loading model data: {str(e)}")
    exit(1)

# Create a new model instance
print("Creating new model instance...")
model = LogisticRegression(
    C=1.0,
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced'
)

# Set model attributes
print("Setting model attributes...")
model.coef_ = np.array([model_data['coefficients']])
model.intercept_ = np.array([model_data['intercept']])
model.classes_ = np.array([0, 1])

# Save the model
print("Saving model...")
import joblib
joblib.dump(model, os.path.join(MODEL_DIR, 'clinical_lr_model.pkl'))
print(f"Model saved to {os.path.join(MODEL_DIR, 'clinical_lr_model.pkl')}")

print("Model fix completed successfully!")
