"""
Simple test script for the hybrid model.
"""
import os
import sys
import traceback

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting simple hybrid model test...")

try:
    print("Importing HybridHeartFailureModel...")
    from hybrid_model import HybridHeartFailureModel
    
    print("Creating test patient data...")
    test_patient = {
        'age': 60,
        'gender': 'Male',
        'blood_pressure': '140/90',
        'cholesterol': 240,
        'fasting_blood_sugar': 130,
        'max_heart_rate': 100,
        'exercise_induced_angina': True,
        'st_depression': 2.0,
        'slope_of_st': 'Downsloping',
        'number_of_major_vessels': 2,
        'thalassemia': 'Reversible Defect',
        'biomarkers': {
            'nt_probnp': 1200
        }
    }
    
    print("Creating hybrid model...")
    hybrid_model = HybridHeartFailureModel()
    
    print("Making prediction...")
    try:
        prediction_result = hybrid_model.predict_heart_failure(test_patient)
        
        print(f"\nHybrid Model Prediction Results:")
        print(f"Overall Probability: {prediction_result['probability']:.4f}")
        print(f"Confidence: {prediction_result['confidence']:.4f}")
        print(f"Risk Level: {prediction_result['risk_level']}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        traceback.print_exc()

except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()

print("Test completed.")
