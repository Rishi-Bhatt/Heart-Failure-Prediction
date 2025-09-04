import requests
import json

# Test data
test_data = {
    'name': 'Test Patient',
    'age': 65,
    'gender': 'Male',
    'blood_pressure': '140/90',
    'cholesterol': 220,
    'fasting_blood_sugar': 110,
    'chest_pain_type': 'Typical Angina',
    'ecg_result': 'Normal',
    'max_heart_rate': 140,
    'exercise_induced_angina': True,
    'st_depression': 1.5,
    'slope_of_st': 'Flat',
    'number_of_major_vessels': 2,
    'thalassemia': 'Normal',
    'prior_cardiac_event': {
        'type': 'Myocardial Infarction',
        'time_since_event': 6,
        'severity': 'Moderate'
    },
    'medications': [
        {
            'type': 'Beta-blockers',
            'time_of_administration': 2
        },
        {
            'type': 'ACE inhibitors',
            'time_of_administration': 4
        }
    ]
}

print("Sending test prediction request...")
try:
    response = requests.post('http://localhost:8000/api/predict', json=test_data)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction successful!")
        print(f"Patient ID: {result.get('patient_id')}")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"ECG signal length: {len(result.get('ecg_signal', []))}")
        print(f"Abnormalities: {json.dumps(result.get('abnormalities', {}), indent=2)}")
    else:
        print("Prediction failed!")
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Error: {str(e)}")

print("Test completed.")
