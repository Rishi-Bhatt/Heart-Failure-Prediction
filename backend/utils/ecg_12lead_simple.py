"""
Simplified 12-Lead ECG Generator

This module provides a streamlined implementation for generating 12-lead ECG data
by extending the existing single-lead ECG functionality.
"""

import numpy as np
import random
from datetime import datetime

# Import existing ECG generator for compatibility
from .ecg_generator import generate_ecg

# Standard lead order for 12-lead ECG
STANDARD_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Lead relationships (based on Einthoven's triangle and precordial lead placement)
LEAD_RELATIONSHIPS = {
    # Limb leads
    'I': {'base': 'II', 'factor': 0.5, 'offset': 0.2},
    'II': {'base': 'II', 'factor': 1.0, 'offset': 0.0},  # Reference lead
    'III': {'base': 'II', 'factor': 0.7, 'offset': -0.1},

    # Augmented limb leads
    'aVR': {'base': 'II', 'factor': -0.5, 'offset': 0.1},
    'aVL': {'base': 'I', 'factor': 0.8, 'offset': 0.0},
    'aVF': {'base': 'III', 'factor': 0.9, 'offset': 0.0},

    # Precordial leads
    'V1': {'base': 'II', 'factor': 0.5, 'offset': -0.3, 'invert_t': True},
    'V2': {'base': 'II', 'factor': 0.7, 'offset': -0.2, 'invert_t': True},
    'V3': {'base': 'II', 'factor': 0.9, 'offset': -0.1},
    'V4': {'base': 'II', 'factor': 1.2, 'offset': 0.0},
    'V5': {'base': 'II', 'factor': 1.0, 'offset': 0.1},
    'V6': {'base': 'I', 'factor': 0.8, 'offset': 0.0},
}

def generate_12lead_ecg(patient_data, base_signal=None, time_array=None):
    """
    Generate a simplified 12-lead ECG based on patient data

    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient clinical data
    base_signal : numpy.ndarray, optional
        Base ECG signal (Lead II) to use instead of generating a new one
    time_array : numpy.ndarray, optional
        Time array corresponding to the base signal

    Returns:
    --------
    ecg_data : dict
        Dictionary containing ECG signals for all 12 leads, time array, and metadata
    """
    # Generate base ECG signal (Lead II) if not provided
    if base_signal is None or time_array is None:
        base_signal, time_array = generate_ecg(patient_data)
        print("Generated new base signal for 12-lead ECG")
    else:
        print("Using provided base signal for 12-lead ECG")

    # Generate all 12 leads based on the base signal
    leads_data = {}
    for lead_name in STANDARD_LEAD_ORDER:
        # Get lead relationship parameters
        relationship = LEAD_RELATIONSHIPS[lead_name]

        # Generate lead signal based on the relationship
        if lead_name == 'II':
            # Lead II is our base signal
            lead_signal = base_signal
        else:
            # Derive other leads from the base signal
            lead_signal = derive_lead(base_signal, relationship, patient_data)

        # Store the lead data
        leads_data[lead_name] = lead_signal.tolist()

    # Create metadata
    metadata = {
        'sampling_rate': len(base_signal) / 10,  # Assuming 10 seconds of data
        'duration': 10,  # seconds
        'heart_rate': calculate_heart_rate(patient_data),
        'patient_age': patient_data.get('age', 60),
        'patient_gender': patient_data.get('gender', 'Male'),
        'timestamp': datetime.now().isoformat()
    }

    # Return complete 12-lead ECG data
    return {
        'leads': leads_data,
        'time': time_array.tolist(),
        'metadata': metadata,
        'lead_order': STANDARD_LEAD_ORDER
    }

def derive_lead(base_signal, relationship, patient_data):
    """
    Derive a lead signal based on its relationship to the base signal

    Parameters:
    -----------
    base_signal : numpy.ndarray
        Base ECG signal (Lead II)
    relationship : dict
        Parameters defining the relationship to the base signal
    patient_data : dict
        Dictionary containing patient clinical data

    Returns:
    --------
    lead_signal : numpy.ndarray
        Derived lead signal
    """
    # Apply scaling factor
    lead_signal = base_signal * relationship['factor']

    # Apply offset
    lead_signal = lead_signal + relationship['offset']

    # Invert T wave if specified
    if relationship.get('invert_t', False):
        lead_signal = invert_t_wave(lead_signal)

    # Apply patient-specific modifications
    lead_signal = apply_patient_modifications(lead_signal, patient_data, relationship)

    # Add some random variation to make it more realistic
    noise = np.random.normal(0, 0.02, len(lead_signal))
    lead_signal = lead_signal + noise

    return lead_signal

def invert_t_wave(signal):
    """
    Invert T waves in the signal

    This is a simplified implementation that approximates T wave inversion
    by detecting peaks and inverting the signal in those regions.

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal

    Returns:
    --------
    modified_signal : numpy.ndarray
        Signal with inverted T waves
    """
    # This is a very simplified approach
    # In a real implementation, we would detect T waves properly

    # Create a copy of the signal
    modified_signal = signal.copy()

    # Find R peaks (simplified)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    # For each peak, invert the T wave region (approximately 200-400ms after R peak)
    for peak in peaks:
        t_start = min(peak + 50, len(signal) - 1)
        t_end = min(peak + 150, len(signal) - 1)

        # Find the T wave peak in this region
        t_peak = t_start
        for i in range(t_start, t_end):
            if abs(signal[i]) > abs(signal[t_peak]):
                t_peak = i

        # Invert the T wave
        for i in range(t_start, t_end):
            # Apply a gradual inversion centered on the T peak
            distance = abs(i - t_peak)
            inversion_factor = max(0, 1 - distance / 50)
            modified_signal[i] = signal[i] - 2 * signal[i] * inversion_factor

    return modified_signal

def apply_patient_modifications(signal, patient_data, relationship):
    """
    Apply patient-specific modifications to the lead signal

    Parameters:
    -----------
    signal : numpy.ndarray
        Lead signal
    patient_data : dict
        Dictionary containing patient clinical data
    relationship : dict
        Parameters defining the relationship to the base signal

    Returns:
    --------
    modified_signal : numpy.ndarray
        Modified lead signal
    """
    # Create a copy of the signal
    modified_signal = signal.copy()

    # Apply modifications based on patient conditions
    lead_name = next((lead for lead, rel in LEAD_RELATIONSHIPS.items() if rel == relationship), None)

    if lead_name:
        # Left ventricular hypertrophy
        if patient_data.get('lvh', False) and lead_name in ['V5', 'V6', 'I', 'aVL']:
            # Increased R wave amplitude in left-sided leads
            modified_signal *= 1.3

        # Right ventricular hypertrophy
        if patient_data.get('rvh', False) and lead_name in ['V1', 'V2', 'aVR']:
            # Increased R wave amplitude in right-sided leads
            modified_signal *= 1.3

        # Myocardial infarction
        if patient_data.get('prior_cardiac_event', {}).get('type', '') == 'Myocardial Infarction':
            location = patient_data.get('prior_cardiac_event', {}).get('location', 'Anterior')

            # Apply lead-specific changes based on MI location
            if (location == 'Anterior' and lead_name in ['V1', 'V2', 'V3', 'V4']) or \
               (location == 'Inferior' and lead_name in ['II', 'III', 'aVF']) or \
               (location == 'Lateral' and lead_name in ['I', 'aVL', 'V5', 'V6']):
                # Simplified MI changes
                modified_signal = add_q_wave(modified_signal)

    return modified_signal

def add_q_wave(signal):
    """
    Add or deepen Q waves to simulate myocardial infarction

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal

    Returns:
    --------
    modified_signal : numpy.ndarray
        Signal with added/deepened Q waves
    """
    # This is a simplified implementation
    # In a real system, we would identify QRS complexes and modify them

    # Create a copy of the signal
    modified_signal = signal.copy()

    # Find R peaks (simplified)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    # For each peak, add or deepen Q wave
    for peak in peaks:
        # Look for the region just before the R peak
        q_start = max(peak - 20, 0)
        q_end = peak

        # Add a Q wave
        for i in range(q_start, q_end):
            # Create a gradual Q wave
            position = (i - q_start) / (q_end - q_start)
            q_factor = np.sin(position * np.pi) * 0.3
            modified_signal[i] = signal[i] - q_factor

    return modified_signal

def calculate_heart_rate(patient_data):
    """
    Calculate heart rate based on patient data
    """
    # Base heart rate
    base_hr = float(patient_data.get('max_heart_rate', 75))

    # Adjust based on age
    age = float(patient_data.get('age', 60))
    age_factor = 1 - (age - 40) * 0.005 if age > 40 else 1

    # Adjust based on exercise induced angina
    angina_factor = 0.9 if patient_data.get('exercise_induced_angina', False) else 1

    # Calculate adjusted heart rate
    adjusted_hr = base_hr * age_factor * angina_factor

    # Ensure heart rate is within physiological limits
    adjusted_hr = max(40, min(adjusted_hr, 200))

    return adjusted_hr

def derive_12lead_from_base(base_signal, time_array, patient_info=None, abnormalities=None):
    """
    Derive a 12-lead ECG from a base signal (Lead II)

    Parameters:
    -----------
    base_signal : numpy.ndarray
        Base ECG signal (Lead II)
    time_array : numpy.ndarray
        Time array corresponding to the base signal
    patient_info : dict, optional
        Dictionary containing patient clinical data
    abnormalities : dict, optional
        Dictionary containing abnormalities detected in the single-lead ECG

    Returns:
    --------
    ecg_data : dict
        Dictionary containing ECG signals for all 12 leads, time array, and metadata
    """
    # Use the existing generate_12lead_ecg function with the provided base signal
    ecg_data = generate_12lead_ecg(patient_info or {}, base_signal, time_array)

    # If abnormalities are provided, ensure they are reflected in the 12-lead ECG
    if abnormalities:
        # Add abnormalities to the ECG data
        ecg_data['abnormalities'] = abnormalities

        # Ensure PVCs are reflected in the appropriate leads
        if 'PVCs' in abnormalities and abnormalities['PVCs']:
            for pvc in abnormalities['PVCs']:
                pvc_time = pvc.get('time', 0)
                pvc_duration = pvc.get('duration', 0.2)

                # Find the sample indices corresponding to the PVC time
                start_idx = int((pvc_time - 0.1) * (len(time_array) / time_array[-1]))
                end_idx = int((pvc_time + pvc_duration) * (len(time_array) / time_array[-1]))

                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, len(time_array) - 1))
                end_idx = max(0, min(end_idx, len(time_array) - 1))

                # Apply PVC morphology to all leads
                for lead_name in ecg_data['leads']:
                    # Get the lead signal
                    lead_signal = np.array(ecg_data['leads'][lead_name])

                    # Apply PVC morphology based on lead
                    if lead_name in ['II', 'V1', 'V2', 'V3']:
                        # Create PVC shape (wider and higher amplitude)
                        pvc_width = end_idx - start_idx
                        pvc_shape = -1.5 * np.sin(np.linspace(0, np.pi, pvc_width))
                        lead_signal[start_idx:end_idx] = pvc_shape
                    elif lead_name in ['V4', 'V5', 'V6']:
                        # Different PVC morphology in lateral leads
                        pvc_width = end_idx - start_idx
                        pvc_shape = -1.2 * np.sin(np.linspace(0, np.pi, pvc_width))
                        lead_signal[start_idx:end_idx] = pvc_shape
                    elif lead_name in ['I', 'aVL']:
                        # Different PVC morphology in high lateral leads
                        pvc_width = end_idx - start_idx
                        pvc_shape = -0.8 * np.sin(np.linspace(0, np.pi, pvc_width))
                        lead_signal[start_idx:end_idx] = pvc_shape

                    # Update the lead signal in the ECG data
                    ecg_data['leads'][lead_name] = lead_signal.tolist()

    return ecg_data
