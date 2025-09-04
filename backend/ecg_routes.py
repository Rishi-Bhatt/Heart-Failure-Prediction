"""
ECG Routes for the Heart Failure Prediction System

This module provides routes for ECG data, including 12-lead ECG visualization.
"""

from flask import jsonify, request
import os
import json
import numpy as np
from datetime import datetime

# Import ECG modules
try:
    from utils.ecg_generator import generate_ecg, analyze_ecg
except ImportError:
    # Fallback functions if the real ones aren't available
    def generate_ecg(patient_data):
        """Generate a simplified ECG signal based on patient data"""
        # Create a simple sine wave as a placeholder
        duration = 10  # seconds
        sampling_rate = 500  # Hz
        num_points = int(duration * sampling_rate)
        time_array = np.linspace(0, duration, num_points)

        # Get heart rate from patient data or use default
        try:
            if isinstance(patient_data, dict):
                heart_rate = float(patient_data.get('max_heart_rate', 75))
            else:
                heart_rate = 75
        except (ValueError, TypeError):
            heart_rate = 75

        # Ensure heart rate is within reasonable bounds
        heart_rate = max(40, min(heart_rate, 200))

        # Calculate frequency based on heart rate
        frequency = heart_rate / 60.0

        # Generate a simple ECG-like signal
        signal = np.zeros(num_points)
        for i in range(num_points):
            t = time_array[i]
            # P wave
            p_wave = 0.25 * np.exp(-((t % (1/frequency) - 0.2) ** 2) / 0.001)
            # QRS complex
            qrs = 1.0 * np.exp(-((t % (1/frequency) - 0.4) ** 2) / 0.0001)
            # T wave
            t_wave = 0.35 * np.exp(-((t % (1/frequency) - 0.6) ** 2) / 0.002)

            signal[i] = p_wave - 0.1 * qrs + qrs + t_wave

        return signal, time_array

    def analyze_ecg(ecg_signal):
        """Analyze ECG signal to detect abnormalities"""
        # Return a placeholder analysis
        return {
            'heart_rate': 75,
            'rhythm': 'Normal Sinus Rhythm',
            'abnormalities': []
        }

def generate_patient_specific_analysis(patient_data):
    """Generate patient-specific ECG analysis based on patient data"""
    # Extract relevant patient information
    patient_info = patient_data.get('patient_data', {})

    # Get basic patient parameters
    age = patient_info.get('age', '60')
    try:
        age = int(age)
    except (ValueError, TypeError):
        age = 60

    gender = patient_info.get('gender', 'Male')

    # Get cardiac-related parameters
    heart_rate_str = patient_info.get('max_heart_rate', '75')
    try:
        heart_rate = int(heart_rate_str)
    except (ValueError, TypeError):
        heart_rate = 75

    blood_pressure = patient_info.get('blood_pressure', '120/80')
    try:
        systolic, diastolic = blood_pressure.split('/')
        systolic = int(systolic)
        diastolic = int(diastolic)
    except (ValueError, TypeError, AttributeError):
        systolic = 120
        diastolic = 80

    cholesterol_str = patient_info.get('cholesterol', '200')
    try:
        cholesterol = int(cholesterol_str)
    except (ValueError, TypeError):
        cholesterol = 200

    ecg_result = patient_info.get('ecg_result', 'Normal')
    st_depression_str = patient_info.get('st_depression', '0')
    try:
        st_depression = float(st_depression_str)
    except (ValueError, TypeError):
        st_depression = 0

    slope_of_st = patient_info.get('slope_of_st', 'Flat')
    exercise_induced_angina = patient_info.get('exercise_induced_angina', False)

    # Get biomarkers
    biomarkers = patient_info.get('biomarkers', {})
    nt_probnp_str = biomarkers.get('nt_probnp', '0')
    try:
        nt_probnp = float(nt_probnp_str)
    except (ValueError, TypeError):
        nt_probnp = 0

    # Get prior cardiac events
    prior_event = patient_info.get('prior_cardiac_event', {})
    prior_event_type = prior_event.get('type', '')
    prior_event_severity = prior_event.get('severity', 'Mild')
    prior_event_time_str = prior_event.get('time_since_event', '12')
    try:
        prior_event_time = float(prior_event_time_str)
    except (ValueError, TypeError):
        prior_event_time = 12

    # Get medications
    medications = patient_info.get('medications', [])
    med_types = [med.get('type', '') for med in medications]

    # Initialize analysis results with default structure
    analysis_results = {
        'abnormalities': {
            'rhythm': [],
            'conduction': [],
            'st_changes': [],
            'chamber_enlargement': [],
            'axis_deviation': [],
            'infarction': [],
            'PVCs': [],
            'QT_prolongation': []
        },
        'rhythm': {'name': 'Normal Sinus Rhythm', 'regularity': 'Regular', 'confidence': 0.9},
        'heart_rate': heart_rate,
        'intervals': {
            'PR': 0.16,
            'QRS': 0.08,
            'QT': 0.38,
            'QTc': 0.41
        },
        'axis': {
            'value': 60,
            'category': 'Normal',
            'confidence': 0.7
        }
    }

    # Add patient-specific abnormalities based on their data

    # 1. Rhythm abnormalities
    if 'Arrhythmia' in prior_event_type:
        # Higher probability of arrhythmia if they have a history
        severity_factor = {'Mild': 0.3, 'Moderate': 0.6, 'Severe': 0.9}.get(prior_event_severity, 0.3)
        time_factor = max(0.2, min(1.0, 1.0 - (prior_event_time / 24.0)))
        arrhythmia_prob = 0.7 * severity_factor * time_factor

        if np.random.random() < arrhythmia_prob:
            if heart_rate > 100:
                analysis_results['rhythm'] = {
                    'name': 'Sinus Tachycardia',
                    'regularity': 'Regular',
                    'confidence': 0.85 + (0.1 * np.random.random())
                }
                analysis_results['abnormalities']['rhythm'].append({
                    'type': 'Sinus Tachycardia',
                    'description': f'Fast heart rate of {heart_rate} bpm',
                    'confidence': 0.85 + (0.1 * np.random.random()),
                    'lead': 'II'
                })
            elif np.random.random() < 0.7:
                analysis_results['rhythm'] = {
                    'name': 'Atrial Fibrillation',
                    'regularity': 'Irregular',
                    'confidence': 0.75 + (0.15 * np.random.random())
                }
                analysis_results['abnormalities']['rhythm'].append({
                    'type': 'Atrial Fibrillation',
                    'description': 'Irregular rhythm with absence of P waves',
                    'confidence': 0.75 + (0.15 * np.random.random()),
                    'lead': 'II'
                })
    elif heart_rate > 100:
        analysis_results['rhythm'] = {
            'name': 'Sinus Tachycardia',
            'regularity': 'Regular',
            'confidence': 0.9
        }
        analysis_results['abnormalities']['rhythm'].append({
            'type': 'Sinus Tachycardia',
            'description': f'Fast heart rate of {heart_rate} bpm',
            'confidence': 0.9,
            'lead': 'II'
        })
    elif heart_rate < 60:
        analysis_results['rhythm'] = {
            'name': 'Sinus Bradycardia',
            'regularity': 'Regular',
            'confidence': 0.9
        }
        analysis_results['abnormalities']['rhythm'].append({
            'type': 'Sinus Bradycardia',
            'description': f'Slow heart rate of {heart_rate} bpm',
            'confidence': 0.9,
            'lead': 'II'
        })
    else:
        # Normal sinus rhythm
        analysis_results['abnormalities']['rhythm'].append({
            'type': 'Sinus Rhythm',
            'description': 'Normal sinus rhythm',
            'confidence': 0.95,
            'lead': 'II'
        })

    # 2. Conduction abnormalities
    if age > 70 or 'Heart Block' in prior_event_type:
        # Higher probability of conduction issues in elderly or with history
        block_prob = 0.4 if age > 70 else 0.0
        if 'Heart Block' in prior_event_type:
            severity_factor = {'Mild': 0.3, 'Moderate': 0.6, 'Severe': 0.9}.get(prior_event_severity, 0.3)
            time_factor = max(0.2, min(1.0, 1.0 - (prior_event_time / 24.0)))
            block_prob += 0.5 * severity_factor * time_factor

        if np.random.random() < block_prob:
            # Determine block type based on severity
            if prior_event_severity == 'Severe' and np.random.random() < 0.4:
                block_type = 'Third-degree AV Block'
                block_desc = 'Complete heart block with independent atrial and ventricular activity'
                analysis_results['intervals']['PR'] = 0.0  # No consistent PR in 3rd degree block
            elif prior_event_severity == 'Moderate' or np.random.random() < 0.6:
                block_type = 'Second-degree AV Block'
                block_desc = 'Intermittent failure of AV conduction'
                analysis_results['intervals']['PR'] = 0.22 + (0.04 * np.random.random())  # Prolonged and variable
            else:
                block_type = 'First-degree AV Block'
                block_desc = 'Delayed AV conduction with prolonged PR interval'
                analysis_results['intervals']['PR'] = 0.22 + (0.06 * np.random.random())  # Prolonged PR

            analysis_results['abnormalities']['conduction'].append({
                'type': block_type,
                'description': block_desc,
                'confidence': 0.75 + (0.15 * np.random.random()),
                'lead': 'II'
            })

    # 3. ST segment changes
    if 'ST-T Wave Abnormality' in ecg_result or st_depression > 0.5:
        # ST depression
        st_prob = 0.7 if 'ST-T Wave Abnormality' in ecg_result else 0.0
        st_prob += min(0.8, st_depression / 5.0)  # Scale with ST depression value

        if np.random.random() < st_prob:
            # Determine which leads show ST depression based on common patterns
            leads = ['V5', 'V6', 'I', 'aVL'] if np.random.random() < 0.6 else ['II', 'III', 'aVF']
            for lead in leads[:2]:  # Add to a couple of leads
                analysis_results['abnormalities']['st_changes'].append({
                    'type': 'ST Depression',
                    'description': f'ST segment depression of {st_depression:.1f} mm',
                    'confidence': 0.8 + (0.15 * np.random.random()),
                    'lead': lead
                })

    # Check for ST elevation (myocardial infarction)
    if 'Myocardial Infarction' in prior_event_type:
        severity_factor = {'Mild': 0.3, 'Moderate': 0.6, 'Severe': 0.9}.get(prior_event_severity, 0.3)
        time_factor = max(0.1, min(0.8, 0.8 - (prior_event_time / 24.0)))  # Less likely as time passes
        mi_prob = 0.6 * severity_factor * time_factor

        if np.random.random() < mi_prob:
            # Determine MI location based on random selection
            mi_location = np.random.choice(['Anterior', 'Inferior', 'Lateral', 'Septal'])

            # Select leads based on MI location
            if mi_location == 'Anterior':
                leads = ['V2', 'V3', 'V4']
            elif mi_location == 'Inferior':
                leads = ['II', 'III', 'aVF']
            elif mi_location == 'Lateral':
                leads = ['I', 'aVL', 'V5', 'V6']
            else:  # Septal
                leads = ['V1', 'V2']

            # Add ST elevation to selected leads
            for lead in leads[:2]:  # Add to a couple of leads
                analysis_results['abnormalities']['st_changes'].append({
                    'type': 'ST Elevation',
                    'description': f'ST segment elevation suggesting {mi_location} wall ischemia/infarction',
                    'confidence': 0.75 + (0.2 * np.random.random()),
                    'lead': lead
                })

            # Also add infarction finding
            analysis_results['abnormalities']['infarction'].append({
                'type': f'{mi_location} Infarction',
                'description': f'Q waves suggesting {mi_location.lower()} wall infarction',
                'confidence': 0.7 + (0.2 * np.random.random()),
                'lead': leads[0]
            })

    # 4. Chamber enlargement
    if systolic > 140 or diastolic > 90 or age > 70:
        # Higher probability of left ventricular hypertrophy with hypertension or age
        lvh_prob = 0.0
        if systolic > 160 or diastolic > 100:
            lvh_prob += 0.5  # Severe hypertension
        elif systolic > 140 or diastolic > 90:
            lvh_prob += 0.3  # Mild-moderate hypertension

        if age > 70:
            lvh_prob += 0.2  # Advanced age

        if np.random.random() < lvh_prob:
            analysis_results['abnormalities']['chamber_enlargement'].append({
                'type': 'Left Ventricular Hypertrophy',
                'description': 'Increased QRS voltage suggesting left ventricular hypertrophy',
                'confidence': 0.8 + (0.15 * np.random.random()),
                'lead': 'V5'
            })

    # Check for left atrial enlargement
    if 'Atrial Fibrillation' in prior_event_type or heart_rate > 100:
        lae_prob = 0.4 if 'Atrial Fibrillation' in prior_event_type else 0.0
        lae_prob += 0.2 if heart_rate > 100 else 0.0

        if np.random.random() < lae_prob:
            analysis_results['abnormalities']['chamber_enlargement'].append({
                'type': 'Left Atrial Enlargement',
                'description': 'P-wave abnormality suggesting left atrial enlargement',
                'confidence': 0.75 + (0.15 * np.random.random()),
                'lead': 'V1'
            })

    # 5. Axis deviation
    if age > 70 or 'Heart Block' in prior_event_type:
        # Left axis deviation more common in elderly
        lad_prob = 0.3 if age > 70 else 0.0
        lad_prob += 0.3 if 'Heart Block' in prior_event_type else 0.0

        if np.random.random() < lad_prob:
            analysis_results['axis']['value'] = -30 - int(30 * np.random.random())
            analysis_results['axis']['category'] = 'Left Axis Deviation'

            analysis_results['abnormalities']['axis_deviation'].append({
                'type': 'Left Axis Deviation',
                'description': f'QRS axis deviation to the left ({analysis_results["axis"]["value"]}°)',
                'confidence': 0.85 + (0.1 * np.random.random()),
                'lead': 'aVF'
            })

    # 6. PVCs (Premature Ventricular Contractions)
    pvc_prob = 0.0
    if 'Arrhythmia' in prior_event_type:
        severity_factor = {'Mild': 0.3, 'Moderate': 0.6, 'Severe': 0.9}.get(prior_event_severity, 0.3)
        pvc_prob += 0.5 * severity_factor

    if heart_rate > 100:
        pvc_prob += 0.2  # More likely with tachycardia

    if age > 60:
        pvc_prob += 0.1  # More likely in older patients

    if np.random.random() < pvc_prob:
        # Add 1-3 PVCs
        num_pvcs = np.random.randint(1, 4)
        for i in range(num_pvcs):
            # Random time between 1-9 seconds
            pvc_time = 1.0 + (8.0 * np.random.random())

            analysis_results['abnormalities']['PVCs'].append({
                'type': 'PVC',
                'description': 'Premature ventricular contraction',
                'confidence': 0.85 + (0.1 * np.random.random()),
                'lead': np.random.choice(['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']),
                'time': pvc_time,
                'duration': 0.2 + (0.2 * np.random.random())
            })

    # 7. QT prolongation
    qt_prob = 0.0

    # Medications that can prolong QT
    qt_prolonging_meds = ['Antiarrhythmic', 'Antipsychotic', 'Antibiotic', 'Antidepressant']
    for med_type in med_types:
        if any(qt_med in med_type for qt_med in qt_prolonging_meds):
            qt_prob += 0.4
            break

    # Electrolyte abnormalities can prolong QT
    if nt_probnp > 300:
        qt_prob += 0.3  # Higher NT-proBNP may indicate heart failure which can affect QT

    if np.random.random() < qt_prob:
        # Calculate prolonged QT interval
        qt_interval = 0.45 + (0.1 * np.random.random())
        analysis_results['intervals']['QT'] = qt_interval
        analysis_results['intervals']['QTc'] = qt_interval * np.sqrt(60.0 / heart_rate)  # Bazett's formula

        analysis_results['abnormalities']['QT_prolongation'].append({
            'type': 'Prolonged QT',
            'description': f'QT interval prolongation ({int(qt_interval*1000)} ms)',
            'confidence': 0.8 + (0.15 * np.random.random()),
            'lead': 'II'
        })

    return analysis_results

def register_ecg_routes(app):
    """Register ECG routes with the Flask app"""

    @app.route('/api/patients/<patient_id>/ecg/12lead', methods=['GET', 'OPTIONS'])
    def get_patient_12lead_ecg(patient_id):
        """Get 12-lead ECG data for a patient"""
        # Handle OPTIONS request for CORS preflight
        if request.method == 'OPTIONS':
            return jsonify({'status': 'ok'})

        try:
            # Check if patient exists
            file_path = f'data/patients/{patient_id}.json'
            if not os.path.exists(file_path):
                return jsonify({'error': f'Patient {patient_id} not found'}), 404

            # Load patient data
            with open(file_path, 'r') as f:
                patient_data = json.load(f)

            # Generate a simplified 12-lead ECG data structure
            try:
                # Check if patient data has ECG signal already
                if 'ecg_signal' in patient_data and 'ecg_time' in patient_data and len(patient_data['ecg_signal']) > 0:
                    # Use existing ECG signal as base
                    print(f"Using existing ECG signal for patient {patient_id}")
                    base_signal = np.array(patient_data['ecg_signal'])
                    time_array = np.array(patient_data['ecg_time'])
                else:
                    # Extract patient data for ECG generation
                    patient_info = patient_data.get('patient_data', {})

                    # Add patient_id to ensure consistent ECG generation
                    patient_info['patient_id'] = patient_id

                    print(f"Generating new ECG signal for patient {patient_id}")
                    # Generate base ECG signal
                    base_signal, time_array = generate_ecg(patient_info)

                    # Save the generated ECG back to the patient file for future consistency
                    patient_data['ecg_signal'] = base_signal.tolist()
                    patient_data['ecg_time'] = time_array.tolist()

                    # Update the patient file
                    with open(file_path, 'w') as f:
                        json.dump(patient_data, f, indent=2)
                    print(f"Updated patient file with generated ECG for {patient_id}")
            except Exception as e:
                print(f"Error generating ECG: {str(e)}")
                # Create fallback data
                base_signal = np.zeros(2500)
                time_array = np.linspace(0, 10, 2500)

            # Try to use the 12-lead ECG generator if available
            try:
                # Extract patient info for ECG generation
                patient_info = patient_data.get('patient_data', {})

                # Import the 12-lead ECG generator
                from utils.ecg_12lead_simple import derive_12lead_from_base

                # Get abnormalities from patient data if available
                abnormalities = patient_data.get('abnormalities', None)

                # Generate 12-lead ECG using the base signal as Lead II
                ecg_data = derive_12lead_from_base(base_signal, time_array, patient_info, abnormalities)

                print(f"Generated 12-lead ECG using enhanced generator for patient {patient_id}")
            except ImportError as e:
                print(f"Error importing 12-lead ECG module: {str(e)}")

                # Fallback to basic lead derivation
                # Extract patient info for ECG generation
                patient_info = patient_data.get('patient_data', {})

                # Create derived leads using Einthoven relationships
                # Lead I = Lead II - Lead III
                lead_III = np.zeros_like(base_signal)

                # Apply some basic transformations to create variation
                for i in range(len(base_signal)):
                    # Create Lead III with phase shift and amplitude change
                    idx = max(0, min(len(base_signal)-1, i - int(len(base_signal)*0.03)))
                    lead_III[i] = base_signal[idx] * 0.8

                # Derive Lead I using Einthoven's law: Lead I = Lead II - Lead III
                lead_I = base_signal - lead_III

                # Derive augmented limb leads
                lead_aVR = -(lead_I + base_signal) / 2  # -0.5 * (I + II)
                lead_aVL = (lead_I - lead_III) / 2      # 0.5 * (I - III)
                lead_aVF = (base_signal + lead_III) / 2  # 0.5 * (II + III)

                # Create precordial leads with progressive changes
                # These transformations simulate the normal progression across the precordial leads
                lead_V1 = base_signal * 0.5 - 0.3
                lead_V2 = base_signal * 0.7 - 0.2
                lead_V3 = base_signal * 0.9 - 0.1
                lead_V4 = base_signal * 1.2
                lead_V5 = base_signal * 1.0 + 0.1
                lead_V6 = lead_I * 0.8

                # Get heart rate from patient data if available
                heart_rate = 75  # Default
                if 'patient_data' in patient_data and 'max_heart_rate' in patient_data['patient_data']:
                    try:
                        heart_rate = float(patient_data['patient_data']['max_heart_rate'])
                    except (ValueError, TypeError):
                        pass

                # Create the ECG data structure
                ecg_data = {
                    'leads': {
                        'I': lead_I.tolist(),
                        'II': base_signal.tolist(),  # Lead II is our base signal
                        'III': lead_III.tolist(),
                        'aVR': lead_aVR.tolist(),
                        'aVL': lead_aVL.tolist(),
                        'aVF': lead_aVF.tolist(),
                        'V1': lead_V1.tolist(),
                        'V2': lead_V2.tolist(),
                        'V3': lead_V3.tolist(),
                        'V4': lead_V4.tolist(),
                        'V5': lead_V5.tolist(),
                        'V6': lead_V6.tolist()
                    },
                    'time': time_array.tolist(),
                    'metadata': {
                        'heart_rate': heart_rate,
                        'paper_speed': 25,
                        'amplitude_scale': 10
                    },
                    'lead_order': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
                }

                print(f"Generated 12-lead ECG using basic derivation for patient {patient_id}")

            # Generate patient-specific ECG analysis
            analysis_results = generate_patient_specific_analysis(patient_data)

            # Return the results
            return jsonify({
                'status': 'success',
                'patient_id': patient_id,
                'time': time_array.tolist(),
                'leads': ecg_data['leads'],
                'metadata': ecg_data['metadata'],
                'analysis': analysis_results
            })
        except Exception as e:
            print(f"Error generating 12-lead ECG: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
