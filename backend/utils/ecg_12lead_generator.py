"""
Enhanced ECG Generator for 12-Lead ECG

This module extends the ECG generation capabilities to support all 12 standard ECG leads
with high clinical accuracy. It maintains compatibility with the existing single-lead system
while providing more comprehensive cardiac electrical activity visualization.

References:
1. Kligfield, P., et al. (2007). Recommendations for the standardization and interpretation 
   of the electrocardiogram. Journal of the American College of Cardiology, 49(10), 1109-1127.
2. Malmivuo, J., & Plonsey, R. (1995). Bioelectromagnetism: Principles and Applications of 
   Bioelectric and Biomagnetic Fields. Oxford University Press.
"""

import numpy as np
import math
from scipy import signal
import random
from datetime import datetime

# Import existing ECG generator for compatibility
from utils.ecg_generator import generate_ecg, analyze_ecg

# Constants for ECG generation
SAMPLING_RATE = 500  # Higher sampling rate for better accuracy
DURATION = 10  # seconds
NUM_SAMPLES = int(DURATION * SAMPLING_RATE)

# Standard ECG paper settings
MM_PER_SEC = 25  # 25 mm/sec is standard ECG paper speed
MM_PER_MV = 10   # 10 mm/mV is standard ECG amplitude

# Lead-specific parameters
LEAD_PARAMETERS = {
    # Limb leads
    'I': {'p_amp': 0.15, 'q_amp': -0.1, 'r_amp': 1.0, 's_amp': -0.2, 't_amp': 0.3, 'baseline': 0},
    'II': {'p_amp': 0.25, 'q_amp': -0.2, 'r_amp': 1.5, 's_amp': -0.3, 't_amp': 0.4, 'baseline': 0},
    'III': {'p_amp': 0.1, 'q_amp': -0.1, 'r_amp': 0.7, 's_amp': -0.2, 't_amp': 0.2, 'baseline': 0},
    
    # Augmented limb leads
    'aVR': {'p_amp': -0.15, 'q_amp': 0.1, 'r_amp': -0.5, 's_amp': 0.05, 't_amp': -0.3, 'baseline': 0},
    'aVL': {'p_amp': 0.05, 'q_amp': -0.1, 'r_amp': 0.5, 's_amp': -0.3, 't_amp': 0.1, 'baseline': 0},
    'aVF': {'p_amp': 0.2, 'q_amp': -0.1, 'r_amp': 1.0, 's_amp': -0.2, 't_amp': 0.3, 'baseline': 0},
    
    # Precordial leads
    'V1': {'p_amp': 0.1, 'q_amp': -0.1, 'r_amp': 0.5, 's_amp': -1.5, 't_amp': -0.2, 'baseline': 0},
    'V2': {'p_amp': 0.15, 'q_amp': -0.1, 'r_amp': 0.8, 's_amp': -2.0, 't_amp': -0.5, 'baseline': 0},
    'V3': {'p_amp': 0.15, 'q_amp': -0.1, 'r_amp': 1.0, 's_amp': -1.0, 't_amp': 0.7, 'baseline': 0},
    'V4': {'p_amp': 0.15, 'q_amp': -0.15, 'r_amp': 1.5, 's_amp': -0.5, 't_amp': 0.8, 'baseline': 0},
    'V5': {'p_amp': 0.15, 'q_amp': -0.2, 'r_amp': 1.5, 's_amp': -0.3, 't_amp': 0.6, 'baseline': 0},
    'V6': {'p_amp': 0.15, 'q_amp': -0.2, 'r_amp': 1.2, 's_amp': -0.2, 't_amp': 0.4, 'baseline': 0},
}

# Standard lead order for 12-lead ECG
STANDARD_LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def generate_12lead_ecg(patient_data, return_single_lead=False):
    """
    Generate a 12-lead ECG signal based on patient data
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient clinical data
    return_single_lead : bool, optional
        If True, returns only lead II for backward compatibility
    
    Returns:
    --------
    ecg_data : dict
        Dictionary containing ECG signals for all 12 leads, time array, and metadata
    """
    # Extract relevant parameters
    age = float(patient_data.get('age', 60))
    gender = patient_data.get('gender', 'Male')
    heart_rate = calculate_heart_rate(patient_data)
    
    # Create time array
    time_array = np.linspace(0, DURATION, NUM_SAMPLES)
    
    # Generate base cardiac cycle
    cardiac_cycle = generate_cardiac_cycle(heart_rate)
    
    # Generate all 12 leads
    leads_data = {}
    for lead_name in STANDARD_LEAD_ORDER:
        # Generate lead-specific signal
        lead_signal = generate_lead_signal(
            cardiac_cycle, 
            lead_name, 
            patient_data, 
            heart_rate
        )
        
        # Apply patient-specific modifications
        lead_signal = apply_patient_modifications(lead_signal, patient_data, lead_name)
        
        # Store the lead data
        leads_data[lead_name] = lead_signal.tolist()
    
    # For backward compatibility
    if return_single_lead:
        # Return only lead II (most commonly used single lead)
        return np.array(leads_data['II']), time_array
    
    # Create metadata
    metadata = {
        'sampling_rate': SAMPLING_RATE,
        'duration': DURATION,
        'heart_rate': heart_rate,
        'paper_speed': MM_PER_SEC,
        'amplitude_scale': MM_PER_MV,
        'patient_age': age,
        'patient_gender': gender,
        'timestamp': datetime.now().isoformat()
    }
    
    # Return complete 12-lead ECG data
    return {
        'leads': leads_data,
        'time': time_array.tolist(),
        'metadata': metadata,
        'lead_order': STANDARD_LEAD_ORDER
    }

def calculate_heart_rate(patient_data):
    """
    Calculate heart rate based on patient data with improved accuracy
    """
    # Base heart rate
    base_hr = float(patient_data.get('max_heart_rate', 75))
    
    # Adjust based on age (using clinical formula)
    age = float(patient_data.get('age', 60))
    age_factor = 1 - (age - 40) * 0.005 if age > 40 else 1
    
    # Adjust based on exercise induced angina
    angina_factor = 0.9 if patient_data.get('exercise_induced_angina', False) else 1
    
    # Adjust based on prior cardiac events
    prior_event = patient_data.get('prior_cardiac_event', {})
    event_type = prior_event.get('type', '')
    event_factor = 1.0
    
    if event_type:
        severity = prior_event.get('severity', 'Moderate')
        time_since = float(prior_event.get('time_since_event', 12))
        
        # More severe and recent events have stronger effect
        if severity == 'Severe':
            event_factor = 0.8 if time_since < 6 else 0.9
        elif severity == 'Moderate':
            event_factor = 0.9 if time_since < 6 else 0.95
        else:  # Mild
            event_factor = 0.95 if time_since < 6 else 0.98
    
    # Calculate adjusted heart rate
    adjusted_hr = base_hr * age_factor * angina_factor * event_factor
    
    # Ensure heart rate is within physiological limits
    adjusted_hr = max(40, min(adjusted_hr, 200))
    
    return adjusted_hr

def generate_cardiac_cycle(heart_rate):
    """
    Generate a single cardiac cycle with accurate timing intervals
    
    Parameters:
    -----------
    heart_rate : float
        Heart rate in beats per minute
    
    Returns:
    --------
    cycle_params : dict
        Parameters defining the cardiac cycle
    """
    # Calculate RR interval in seconds
    rr_interval = 60.0 / heart_rate
    
    # Calculate physiologically accurate intervals (in seconds)
    # These follow standard clinical ECG relationships
    pr_interval = 0.16 + random.uniform(-0.02, 0.02)  # Normal: 0.12-0.20s
    qrs_duration = 0.08 + random.uniform(-0.01, 0.01)  # Normal: 0.06-0.10s
    qt_interval = calculate_qt_interval(heart_rate)  # Rate-dependent
    
    # P wave parameters
    p_duration = 0.08 + random.uniform(-0.01, 0.01)  # Normal: 0.07-0.11s
    p_onset = 0
    p_offset = p_onset + p_duration
    
    # QRS complex parameters
    q_onset = p_offset + (pr_interval - p_duration)  # PR interval includes P wave
    r_peak = q_onset + qrs_duration * 0.4
    s_end = q_onset + qrs_duration
    
    # T wave parameters
    t_onset = s_end + 0.05  # ST segment
    t_peak = t_onset + (qt_interval - qrs_duration - 0.05) * 0.5
    t_offset = q_onset + qt_interval
    
    # Ensure T wave ends before the next cycle
    t_offset = min(t_offset, rr_interval - 0.05)
    
    return {
        'rr_interval': rr_interval,
        'pr_interval': pr_interval,
        'qrs_duration': qrs_duration,
        'qt_interval': qt_interval,
        'p_onset': p_onset,
        'p_offset': p_offset,
        'q_onset': q_onset,
        'r_peak': r_peak,
        's_end': s_end,
        't_onset': t_onset,
        't_peak': t_peak,
        't_offset': t_offset
    }

def calculate_qt_interval(heart_rate):
    """
    Calculate QT interval based on heart rate using Bazett's formula
    
    Parameters:
    -----------
    heart_rate : float
        Heart rate in beats per minute
    
    Returns:
    --------
    qt_interval : float
        QT interval in seconds
    """
    # Base QTc (corrected QT) interval
    qtc = 0.41 + random.uniform(-0.02, 0.02)  # Normal: 0.35-0.44s
    
    # Calculate RR interval in seconds
    rr = 60.0 / heart_rate
    
    # Apply Bazett's formula: QT = QTc * sqrt(RR)
    qt = qtc * math.sqrt(rr)
    
    return qt

def generate_lead_signal(cycle_params, lead_name, patient_data, heart_rate):
    """
    Generate ECG signal for a specific lead
    
    Parameters:
    -----------
    cycle_params : dict
        Parameters defining the cardiac cycle
    lead_name : str
        Name of the lead to generate
    patient_data : dict
        Dictionary containing patient clinical data
    heart_rate : float
        Heart rate in beats per minute
    
    Returns:
    --------
    lead_signal : numpy.ndarray
        ECG signal for the specified lead
    """
    # Get lead-specific parameters
    lead_params = LEAD_PARAMETERS[lead_name].copy()
    
    # Adjust parameters based on patient data
    lead_params = adjust_lead_parameters(lead_params, patient_data, lead_name)
    
    # Create empty signal array
    signal_array = np.zeros(NUM_SAMPLES)
    
    # Calculate number of beats
    rr_interval = cycle_params['rr_interval']
    num_beats = int(DURATION / rr_interval) + 1
    
    # Generate each beat
    for beat in range(num_beats):
        # Calculate beat offset in samples
        beat_offset = int(beat * rr_interval * SAMPLING_RATE)
        
        # Check if beat fits within the signal
        if beat_offset >= NUM_SAMPLES:
            continue
        
        # Generate PQRST complex for this beat
        beat_signal = generate_pqrst_complex(cycle_params, lead_params)
        
        # Calculate how much of the beat fits in the signal
        remaining_samples = NUM_SAMPLES - beat_offset
        beat_length = min(len(beat_signal), remaining_samples)
        
        # Add beat to signal
        signal_array[beat_offset:beat_offset + beat_length] += beat_signal[:beat_length]
    
    # Add baseline wander (respiratory variation)
    signal_array = add_baseline_wander(signal_array, lead_params['baseline'])
    
    # Add noise
    signal_array = add_noise(signal_array, 0.01)  # 1% noise
    
    return signal_array

def adjust_lead_parameters(lead_params, patient_data, lead_name):
    """
    Adjust lead parameters based on patient data
    
    Parameters:
    -----------
    lead_params : dict
        Base parameters for the lead
    patient_data : dict
        Dictionary containing patient clinical data
    lead_name : str
        Name of the lead
    
    Returns:
    --------
    adjusted_params : dict
        Adjusted parameters for the lead
    """
    # Create a copy of the parameters
    params = lead_params.copy()
    
    # Extract relevant patient data
    age = float(patient_data.get('age', 60))
    gender = patient_data.get('gender', 'Male')
    
    # Age-related adjustments
    if age > 60:
        # Reduced R wave amplitude in older patients
        params['r_amp'] *= (1.0 - (age - 60) * 0.005)
        
        # Increased Q wave in older patients
        params['q_amp'] *= (1.0 + (age - 60) * 0.003)
    
    # Gender-related adjustments
    if gender == 'Female':
        # Females typically have slightly higher T wave amplitude
        params['t_amp'] *= 1.1
    
    # Condition-specific adjustments
    if patient_data.get('prior_cardiac_event', {}).get('type', '') == 'Myocardial Infarction':
        # Adjust for MI based on location
        location = patient_data.get('prior_cardiac_event', {}).get('location', 'Anterior')
        
        if location == 'Anterior' and lead_name in ['V1', 'V2', 'V3', 'V4']:
            # Anterior MI affects precordial leads
            params['q_amp'] *= 2.0  # Deeper Q waves
            params['r_amp'] *= 0.7  # Reduced R waves
            params['t_amp'] *= -1.2  # Inverted T waves
            
        elif location == 'Inferior' and lead_name in ['II', 'III', 'aVF']:
            # Inferior MI affects inferior leads
            params['q_amp'] *= 2.0
            params['r_amp'] *= 0.7
            params['t_amp'] *= -1.2
            
        elif location == 'Lateral' and lead_name in ['I', 'aVL', 'V5', 'V6']:
            # Lateral MI affects lateral leads
            params['q_amp'] *= 2.0
            params['r_amp'] *= 0.7
            params['t_amp'] *= -1.2
    
    # ST depression/elevation
    st_depression = float(patient_data.get('st_depression', 0))
    if st_depression > 0:
        # Apply ST depression to appropriate leads
        if lead_name in ['V4', 'V5', 'V6', 'II']:
            params['baseline'] -= st_depression * 0.1  # Convert mm to mV
    
    return params

def generate_pqrst_complex(cycle_params, lead_params):
    """
    Generate a single PQRST complex for a specific lead
    
    Parameters:
    -----------
    cycle_params : dict
        Parameters defining the cardiac cycle
    lead_params : dict
        Parameters specific to the lead
    
    Returns:
    --------
    complex_signal : numpy.ndarray
        Signal for a single PQRST complex
    """
    # Calculate number of samples in a cardiac cycle
    cycle_samples = int(cycle_params['rr_interval'] * SAMPLING_RATE)
    
    # Create empty signal array
    complex_signal = np.zeros(cycle_samples)
    
    # Generate P wave (asymmetric gaussian)
    p_onset_idx = int(cycle_params['p_onset'] * SAMPLING_RATE)
    p_offset_idx = int(cycle_params['p_offset'] * SAMPLING_RATE)
    p_duration_samples = p_offset_idx - p_onset_idx
    
    if p_duration_samples > 0:
        p_wave = asymmetric_gaussian(
            p_duration_samples, 
            lead_params['p_amp'], 
            0.6  # P wave asymmetry (slightly right-skewed)
        )
        
        # Add P wave to signal
        for i in range(min(p_duration_samples, len(complex_signal) - p_onset_idx)):
            complex_signal[p_onset_idx + i] += p_wave[i]
    
    # Generate QRS complex
    q_onset_idx = int(cycle_params['q_onset'] * SAMPLING_RATE)
    r_peak_idx = int(cycle_params['r_peak'] * SAMPLING_RATE)
    s_end_idx = int(cycle_params['s_end'] * SAMPLING_RATE)
    
    # Q wave (negative gaussian)
    q_duration_samples = r_peak_idx - q_onset_idx
    if q_duration_samples > 0:
        q_wave = -1 * gaussian(q_duration_samples, abs(lead_params['q_amp']))
        
        # Add Q wave to signal
        for i in range(min(q_duration_samples, len(complex_signal) - q_onset_idx)):
            complex_signal[q_onset_idx + i] += q_wave[i]
    
    # R wave (positive gaussian)
    r_duration_samples = (r_peak_idx - q_onset_idx) * 2
    if r_duration_samples > 0:
        r_wave = gaussian(r_duration_samples, lead_params['r_amp'])
        r_center = r_duration_samples // 2
        
        # Add R wave to signal
        for i in range(min(r_duration_samples, len(complex_signal) - (q_onset_idx + q_duration_samples - r_center))):
            idx = q_onset_idx + q_duration_samples - r_center + i
            if 0 <= idx < len(complex_signal):
                complex_signal[idx] += r_wave[i]
    
    # S wave (negative gaussian)
    s_duration_samples = s_end_idx - r_peak_idx
    if s_duration_samples > 0:
        s_wave = gaussian(s_duration_samples, abs(lead_params['s_amp']))
        
        # Add S wave to signal (inverted)
        for i in range(min(s_duration_samples, len(complex_signal) - r_peak_idx)):
            complex_signal[r_peak_idx + i] -= s_wave[i]
    
    # Generate T wave (asymmetric gaussian)
    t_onset_idx = int(cycle_params['t_onset'] * SAMPLING_RATE)
    t_offset_idx = int(cycle_params['t_offset'] * SAMPLING_RATE)
    t_duration_samples = t_offset_idx - t_onset_idx
    
    if t_duration_samples > 0:
        t_wave = asymmetric_gaussian(
            t_duration_samples, 
            lead_params['t_amp'], 
            0.7  # T wave asymmetry (right-skewed)
        )
        
        # Add T wave to signal
        for i in range(min(t_duration_samples, len(complex_signal) - t_onset_idx)):
            complex_signal[t_onset_idx + i] += t_wave[i]
    
    return complex_signal

def gaussian(length, amplitude, center=None, sigma=None):
    """
    Generate a Gaussian curve
    
    Parameters:
    -----------
    length : int
        Length of the curve in samples
    amplitude : float
        Amplitude of the curve
    center : float, optional
        Center of the curve (default: middle of the curve)
    sigma : float, optional
        Standard deviation (default: 1/6 of the length)
    
    Returns:
    --------
    curve : numpy.ndarray
        Gaussian curve
    """
    if center is None:
        center = length / 2
    if sigma is None:
        sigma = length / 6
    
    x = np.arange(length)
    curve = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    return curve

def asymmetric_gaussian(length, amplitude, asymmetry=0.5):
    """
    Generate an asymmetric Gaussian curve
    
    Parameters:
    -----------
    length : int
        Length of the curve in samples
    amplitude : float
        Amplitude of the curve
    asymmetry : float, optional
        Asymmetry factor (0.5 = symmetric, <0.5 = left skewed, >0.5 = right skewed)
    
    Returns:
    --------
    curve : numpy.ndarray
        Asymmetric Gaussian curve
    """
    center = length * asymmetry
    sigma_left = center / 3
    sigma_right = (length - center) / 3
    
    x = np.arange(length)
    curve = np.zeros(length)
    
    # Left side of the curve
    left_idx = x < center
    curve[left_idx] = amplitude * np.exp(-((x[left_idx] - center) ** 2) / (2 * sigma_left ** 2))
    
    # Right side of the curve
    right_idx = x >= center
    curve[right_idx] = amplitude * np.exp(-((x[right_idx] - center) ** 2) / (2 * sigma_right ** 2))
    
    return curve

def add_baseline_wander(signal, baseline_offset=0, respiratory_rate=15):
    """
    Add baseline wander to simulate respiratory influence
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    baseline_offset : float, optional
        DC offset to add to the signal
    respiratory_rate : float, optional
        Respiratory rate in breaths per minute
    
    Returns:
    --------
    modified_signal : numpy.ndarray
        Signal with baseline wander
    """
    # Calculate respiratory frequency in Hz
    resp_freq = respiratory_rate / 60
    
    # Create time array
    t = np.arange(len(signal)) / SAMPLING_RATE
    
    # Generate baseline wander
    wander = 0.05 * np.sin(2 * np.pi * resp_freq * t) + baseline_offset
    
    # Add wander to signal
    return signal + wander

def add_noise(signal, noise_level=0.01):
    """
    Add realistic noise to the ECG signal
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    noise_level : float, optional
        Noise level as a fraction of signal amplitude
    
    Returns:
    --------
    noisy_signal : numpy.ndarray
        Signal with added noise
    """
    # Calculate signal amplitude
    signal_amplitude = np.max(signal) - np.min(signal)
    
    # Generate white noise
    white_noise = np.random.normal(0, noise_level * signal_amplitude, len(signal))
    
    # Generate power line interference (50/60 Hz)
    t = np.arange(len(signal)) / SAMPLING_RATE
    line_freq = 60  # Hz (North America standard)
    line_noise = 0.01 * signal_amplitude * np.sin(2 * np.pi * line_freq * t)
    
    # Add noise to signal
    return signal + white_noise + line_noise

def apply_patient_modifications(signal, patient_data, lead_name):
    """
    Apply patient-specific modifications to the ECG signal
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    patient_data : dict
        Dictionary containing patient clinical data
    lead_name : str
        Name of the lead
    
    Returns:
    --------
    modified_signal : numpy.ndarray
        Modified ECG signal
    """
    modified_signal = signal.copy()
    
    # Apply modifications based on patient conditions
    
    # Left ventricular hypertrophy
    if patient_data.get('lvh', False):
        if lead_name in ['V5', 'V6', 'I', 'aVL']:
            # Increased R wave amplitude in left-sided leads
            modified_signal *= 1.3
    
    # Right ventricular hypertrophy
    if patient_data.get('rvh', False):
        if lead_name in ['V1', 'V2', 'aVR']:
            # Increased R wave amplitude in right-sided leads
            modified_signal *= 1.3
    
    # Bundle branch blocks
    if patient_data.get('lbbb', False):
        if lead_name in ['V5', 'V6', 'I', 'aVL']:
            # Widen QRS complex for LBBB
            modified_signal = widen_qrs(modified_signal, 1.5)
    
    if patient_data.get('rbbb', False):
        if lead_name in ['V1', 'V2', 'V3']:
            # Add RSR' pattern for RBBB
            modified_signal = add_rsr_prime(modified_signal)
    
    # Atrial fibrillation
    if patient_data.get('afib', False):
        # Remove P waves and add irregular rhythm
        modified_signal = simulate_afib(modified_signal)
    
    return modified_signal

def widen_qrs(signal, factor=1.5):
    """
    Widen QRS complexes to simulate conduction delays
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    factor : float, optional
        Factor by which to widen the QRS complex
    
    Returns:
    --------
    modified_signal : numpy.ndarray
        Signal with widened QRS complexes
    """
    # This is a simplified implementation
    # In a real system, we would identify QRS complexes and resample them
    
    # For now, we'll use a simple low-pass filter to approximate widening
    cutoff = 40 / (factor * 2)  # Hz
    b, a = signal.butter(4, cutoff / (SAMPLING_RATE / 2), 'low')
    return signal.filtfilt(b, a, signal)

def add_rsr_prime(signal):
    """
    Add RSR' pattern to simulate right bundle branch block
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    
    Returns:
    --------
    modified_signal : numpy.ndarray
        Signal with RSR' pattern
    """
    # This is a placeholder for a more complex implementation
    # In a real system, we would identify R waves and modify them
    
    return signal

def simulate_afib(signal):
    """
    Simulate atrial fibrillation by removing P waves and adding irregular rhythm
    
    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    
    Returns:
    --------
    modified_signal : numpy.ndarray
        Signal with atrial fibrillation characteristics
    """
    # This is a placeholder for a more complex implementation
    # In a real system, we would identify P waves and replace them with fibrillatory waves
    
    return signal

# Backward compatibility function
def generate_ecg_enhanced(patient_data):
    """
    Enhanced ECG generator that maintains backward compatibility
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient clinical data
    
    Returns:
    --------
    ecg_signal : numpy.ndarray
        ECG signal (lead II)
    ecg_time : numpy.ndarray
        Time array
    """
    # Use the existing function for backward compatibility
    return generate_ecg(patient_data)
