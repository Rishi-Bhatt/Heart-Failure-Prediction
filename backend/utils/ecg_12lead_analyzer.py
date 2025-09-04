"""
Simplified 12-Lead ECG Analyzer

This module provides a streamlined implementation for analyzing 12-lead ECG data
and detecting abnormalities across all leads.
"""

import numpy as np
from .ecg_analyzer import detect_abnormalities

# Lead groups for specific abnormality detection
LEAD_GROUPS = {
    'anterior': ['V1', 'V2', 'V3', 'V4'],
    'lateral': ['I', 'aVL', 'V5', 'V6'],
    'inferior': ['II', 'III', 'aVF'],
    'septal': ['V1', 'V2'],
    'right_ventricular': ['V1', 'V2', 'V3R', 'V4R'],  # Right-sided leads (not all included in standard 12-lead)
}

def analyze_12lead_ecg(ecg_data):
    """
    Analyze 12-lead ECG data to detect abnormalities

    Parameters:
    -----------
    ecg_data : dict
        Dictionary containing 12-lead ECG data

    Returns:
    --------
    analysis_results : dict
        Dictionary containing analysis results and detected abnormalities
    """
    # Extract data
    leads_data = ecg_data['leads']
    time_array = np.array(ecg_data['time'])
    lead_order = ecg_data.get('lead_order', list(leads_data.keys()))

    # Initialize results
    abnormalities = {
        'rhythm': [],
        'conduction': [],
        'st_changes': [],
        'chamber_enlargement': [],
        'axis_deviation': [],
        'infarction': [],
        'PVCs': [],
        'QT_prolongation': []
    }

    # Analyze each lead individually
    lead_analyses = {}
    for lead_name in lead_order:
        if lead_name in leads_data:
            # Convert lead data to numpy array
            lead_signal = np.array(leads_data[lead_name])

            # Use existing abnormality detection for basic abnormalities
            lead_abnormalities = detect_abnormalities(lead_signal, time_array)

            # Store lead-specific analysis
            lead_analyses[lead_name] = lead_abnormalities

            # Add abnormalities to the overall results
            for abnormality_type, instances in lead_abnormalities.items():
                if instances:
                    for instance in instances:
                        # Add lead information to the abnormality
                        instance['lead'] = lead_name
                        abnormalities[abnormality_type].append(instance)

    # Perform 12-lead specific analyses
    detect_axis_deviation(leads_data, abnormalities)
    detect_chamber_enlargement(leads_data, abnormalities)
    detect_infarction(leads_data, abnormalities)
    detect_st_changes(leads_data, abnormalities)

    # Create comprehensive analysis results
    analysis_results = {
        'abnormalities': abnormalities,
        'lead_analyses': lead_analyses,
        'rhythm': determine_rhythm(abnormalities),
        'heart_rate': estimate_heart_rate(leads_data['II'], time_array),
        'intervals': measure_intervals(leads_data['II'], time_array),
        'axis': calculate_axis(leads_data),
    }

    return analysis_results

def detect_axis_deviation(leads_data, abnormalities):
    """
    Detect cardiac axis deviation

    Parameters:
    -----------
    leads_data : dict
        Dictionary containing lead signals
    abnormalities : dict
        Dictionary to store detected abnormalities
    """
    # This is a simplified implementation
    # In a real system, we would calculate the actual cardiac axis

    # Check for left axis deviation (LAD)
    if 'I' in leads_data and 'aVF' in leads_data:
        lead_I = np.array(leads_data['I'])
        lead_aVF = np.array(leads_data['aVF'])

        # Simplified check: positive I and negative aVF suggests LAD
        if np.max(lead_I) > 0.5 and np.min(lead_aVF) < -0.2:
            abnormalities['axis_deviation'].append({
                'type': 'Left Axis Deviation',
                'description': 'Cardiac axis between -30° and -90°',
                'confidence': 0.8,
                'leads': ['I', 'aVF']
            })

    # Check for right axis deviation (RAD)
    if 'I' in leads_data and 'aVF' in leads_data:
        lead_I = np.array(leads_data['I'])
        lead_aVF = np.array(leads_data['aVF'])

        # Simplified check: negative I and positive aVF suggests RAD
        if np.min(lead_I) < -0.2 and np.max(lead_aVF) > 0.5:
            abnormalities['axis_deviation'].append({
                'type': 'Right Axis Deviation',
                'description': 'Cardiac axis between +90° and +180°',
                'confidence': 0.8,
                'leads': ['I', 'aVF']
            })

def detect_chamber_enlargement(leads_data, abnormalities):
    """
    Detect chamber enlargement

    Parameters:
    -----------
    leads_data : dict
        Dictionary containing lead signals
    abnormalities : dict
        Dictionary to store detected abnormalities
    """
    # Check for left ventricular hypertrophy (LVH)
    if 'V5' in leads_data and 'V6' in leads_data:
        lead_V5 = np.array(leads_data['V5'])
        lead_V6 = np.array(leads_data['V6'])

        # Simplified Sokolow-Lyon criteria: S in V1 + R in V5 or V6 > 35 mm
        r_wave_v5 = np.max(lead_V5)
        r_wave_v6 = np.max(lead_V6)

        if r_wave_v5 > 2.5 or r_wave_v6 > 2.5:  # 25 mm in 10 mm/mV scale
            abnormalities['chamber_enlargement'].append({
                'type': 'Left Ventricular Hypertrophy',
                'description': 'Increased R wave amplitude in left precordial leads',
                'confidence': 0.7,
                'leads': ['V5', 'V6']
            })

    # Check for right ventricular hypertrophy (RVH)
    if 'V1' in leads_data and 'V2' in leads_data:
        lead_V1 = np.array(leads_data['V1'])
        lead_V2 = np.array(leads_data['V2'])

        # Simplified criteria: R in V1 > 7 mm or R/S ratio in V1 > 1
        r_wave_v1 = np.max(lead_V1)
        s_wave_v1 = abs(np.min(lead_V1))

        if r_wave_v1 > 0.7 or (s_wave_v1 > 0 and r_wave_v1 / s_wave_v1 > 1):
            abnormalities['chamber_enlargement'].append({
                'type': 'Right Ventricular Hypertrophy',
                'description': 'Increased R wave amplitude in right precordial leads',
                'confidence': 0.7,
                'leads': ['V1', 'V2']
            })

def detect_infarction(leads_data, abnormalities):
    """
    Detect myocardial infarction patterns

    Parameters:
    -----------
    leads_data : dict
        Dictionary containing lead signals
    abnormalities : dict
        Dictionary to store detected abnormalities
    """
    # Check for anterior MI
    anterior_q_waves = 0
    for lead in LEAD_GROUPS['anterior']:
        if lead in leads_data:
            lead_signal = np.array(leads_data[lead])
            if has_pathological_q_wave(lead_signal):
                anterior_q_waves += 1

    if anterior_q_waves >= 2:
        abnormalities['infarction'].append({
            'type': 'Anterior Myocardial Infarction',
            'description': 'Pathological Q waves in anterior leads',
            'confidence': 0.8,
            'leads': LEAD_GROUPS['anterior']
        })

    # Check for inferior MI
    inferior_q_waves = 0
    for lead in LEAD_GROUPS['inferior']:
        if lead in leads_data:
            lead_signal = np.array(leads_data[lead])
            if has_pathological_q_wave(lead_signal):
                inferior_q_waves += 1

    if inferior_q_waves >= 2:
        abnormalities['infarction'].append({
            'type': 'Inferior Myocardial Infarction',
            'description': 'Pathological Q waves in inferior leads',
            'confidence': 0.8,
            'leads': LEAD_GROUPS['inferior']
        })

    # Check for lateral MI
    lateral_q_waves = 0
    for lead in LEAD_GROUPS['lateral']:
        if lead in leads_data:
            lead_signal = np.array(leads_data[lead])
            if has_pathological_q_wave(lead_signal):
                lateral_q_waves += 1

    if lateral_q_waves >= 2:
        abnormalities['infarction'].append({
            'type': 'Lateral Myocardial Infarction',
            'description': 'Pathological Q waves in lateral leads',
            'confidence': 0.8,
            'leads': LEAD_GROUPS['lateral']
        })

def has_pathological_q_wave(signal):
    """
    Check if a signal has pathological Q waves

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal

    Returns:
    --------
    has_q_wave : bool
        True if pathological Q waves are detected
    """
    # This is a simplified implementation
    # In a real system, we would use more sophisticated detection

    # Find R peaks (simplified)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    # Check for Q waves before R peaks
    for peak in peaks:
        # Look for the region just before the R peak
        q_start = max(peak - 40, 0)
        q_end = peak

        # Find the minimum value in this region
        q_min = np.min(signal[q_start:q_end])

        # Check if it's a significant Q wave (simplified criteria)
        if q_min < -0.2:  # 2 mm in 10 mm/mV scale
            return True

    return False

def detect_st_changes(leads_data, abnormalities):
    """
    Detect ST segment changes

    Parameters:
    -----------
    leads_data : dict
        Dictionary containing lead signals
    abnormalities : dict
        Dictionary to store detected abnormalities
    """
    # Check for ST elevation
    for lead_group_name, lead_group in LEAD_GROUPS.items():
        st_elevations = 0
        affected_leads = []

        for lead in lead_group:
            if lead in leads_data:
                lead_signal = np.array(leads_data[lead])
                if has_st_elevation(lead_signal):
                    st_elevations += 1
                    affected_leads.append(lead)

        if st_elevations >= 2:
            abnormalities['st_changes'].append({
                'type': f'ST Elevation in {lead_group_name.replace("_", " ").title()} Leads',
                'description': 'ST segment elevation ≥ 1 mm in two or more contiguous leads',
                'confidence': 0.8,
                'leads': affected_leads
            })

    # Check for ST depression
    for lead_group_name, lead_group in LEAD_GROUPS.items():
        st_depressions = 0
        affected_leads = []

        for lead in lead_group:
            if lead in leads_data:
                lead_signal = np.array(leads_data[lead])
                if has_st_depression(lead_signal):
                    st_depressions += 1
                    affected_leads.append(lead)

        if st_depressions >= 2:
            abnormalities['st_changes'].append({
                'type': f'ST Depression in {lead_group_name.replace("_", " ").title()} Leads',
                'description': 'ST segment depression ≥ 0.5 mm in two or more contiguous leads',
                'confidence': 0.8,
                'leads': affected_leads
            })

def has_st_elevation(signal):
    """
    Check if a signal has ST segment elevation

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal

    Returns:
    --------
    has_elevation : bool
        True if ST elevation is detected
    """
    # This is a simplified implementation
    # In a real system, we would use more sophisticated detection

    # Find R peaks (simplified)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    # Check for ST elevation after R peaks
    for peak in peaks:
        # Look for the ST segment (80-120 ms after the R peak)
        st_start = min(peak + 40, len(signal) - 1)
        st_end = min(peak + 60, len(signal) - 1)

        # Calculate the average ST level
        st_level = np.mean(signal[st_start:st_end])

        # Check if it's elevated (simplified criteria)
        if st_level > 0.1:  # 1 mm in 10 mm/mV scale
            return True

    return False

def has_st_depression(signal):
    """
    Check if a signal has ST segment depression

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal

    Returns:
    --------
    has_depression : bool
        True if ST depression is detected
    """
    # This is a simplified implementation
    # In a real system, we would use more sophisticated detection

    # Find R peaks (simplified)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    # Check for ST depression after R peaks
    for peak in peaks:
        # Look for the ST segment (80-120 ms after the R peak)
        st_start = min(peak + 40, len(signal) - 1)
        st_end = min(peak + 60, len(signal) - 1)

        # Calculate the average ST level
        st_level = np.mean(signal[st_start:st_end])

        # Check if it's depressed (simplified criteria)
        if st_level < -0.05:  # 0.5 mm in 10 mm/mV scale
            return True

    return False

def determine_rhythm(abnormalities):
    """
    Determine the overall cardiac rhythm

    Parameters:
    -----------
    abnormalities : dict
        Dictionary containing detected abnormalities

    Returns:
    --------
    rhythm : dict
        Dictionary containing rhythm information
    """
    # Check for specific rhythm abnormalities
    if any(a['type'] == 'Atrial Fibrillation' for a in abnormalities.get('rhythm', [])):
        return {
            'name': 'Atrial Fibrillation',
            'regularity': 'Irregularly Irregular',
            'confidence': 0.8
        }

    if any(a['type'] == 'Atrial Flutter' for a in abnormalities.get('rhythm', [])):
        return {
            'name': 'Atrial Flutter',
            'regularity': 'Regular',
            'confidence': 0.8
        }

    if any(a['type'] == 'Ventricular Tachycardia' for a in abnormalities.get('rhythm', [])):
        return {
            'name': 'Ventricular Tachycardia',
            'regularity': 'Regular',
            'confidence': 0.9
        }

    # Default to normal sinus rhythm
    return {
        'name': 'Normal Sinus Rhythm',
        'regularity': 'Regular',
        'confidence': 0.9
    }

def estimate_heart_rate(signal, time_array):
    """
    Estimate heart rate from the ECG signal

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    time_array : numpy.ndarray
        Time array

    Returns:
    --------
    heart_rate : float
        Estimated heart rate in beats per minute
    """
    # This is a simplified implementation
    # In a real system, we would use more sophisticated detection

    # Find R peaks (simplified)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > 0.5:
            peaks.append(i)

    # Calculate average RR interval
    if len(peaks) < 2:
        return 75  # Default heart rate

    rr_intervals = []
    for i in range(1, len(peaks)):
        rr_interval = time_array[peaks[i]] - time_array[peaks[i-1]]
        rr_intervals.append(rr_interval)

    avg_rr_interval = np.mean(rr_intervals)

    # Convert to heart rate
    heart_rate = 60 / avg_rr_interval

    return heart_rate

def measure_intervals(signal, time_array):
    """
    Measure key ECG intervals

    Parameters:
    -----------
    signal : numpy.ndarray
        ECG signal
    time_array : numpy.ndarray
        Time array

    Returns:
    --------
    intervals : dict
        Dictionary containing interval measurements
    """
    # This is a simplified implementation
    # In a real system, we would use more sophisticated detection

    # Default values
    intervals = {
        'PR': 0.16,  # seconds
        'QRS': 0.08,  # seconds
        'QT': 0.38,  # seconds
        'QTc': 0.41,  # seconds
    }

    # Calculate QTc using Bazett's formula
    heart_rate = estimate_heart_rate(signal, time_array)
    rr_interval = 60 / heart_rate
    intervals['QTc'] = intervals['QT'] / np.sqrt(rr_interval)

    return intervals

def calculate_axis(leads_data):
    """
    Calculate the cardiac axis

    Parameters:
    -----------
    leads_data : dict
        Dictionary containing lead signals

    Returns:
    --------
    axis : dict
        Dictionary containing axis information
    """
    # This is a simplified implementation
    # In a real system, we would calculate the actual cardiac axis

    # Check if we have the necessary leads
    if 'I' not in leads_data or 'aVF' not in leads_data:
        return {
            'value': 60,  # Default normal axis
            'category': 'Normal',
            'confidence': 0.5
        }

    # Get lead signals
    lead_I = np.array(leads_data['I'])
    lead_aVF = np.array(leads_data['aVF'])

    # Calculate R wave amplitudes
    r_wave_I = np.max(lead_I)
    r_wave_aVF = np.max(lead_aVF)

    # Simplified axis calculation
    if r_wave_I > 0 and r_wave_aVF > 0:
        # Normal axis (0° to +90°)
        axis_value = 60
        category = 'Normal'
    elif r_wave_I > 0 and r_wave_aVF < 0:
        # Left axis deviation (-30° to -90°)
        axis_value = -45
        category = 'Left Axis Deviation'
    elif r_wave_I < 0 and r_wave_aVF > 0:
        # Right axis deviation (+90° to +180°)
        axis_value = 120
        category = 'Right Axis Deviation'
    else:
        # Extreme axis deviation (-90° to -180°)
        axis_value = -135
        category = 'Extreme Axis Deviation'

    return {
        'value': axis_value,
        'category': category,
        'confidence': 0.7
    }
