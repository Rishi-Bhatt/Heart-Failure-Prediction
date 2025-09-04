import numpy as np

def generate_simple_ecg(patient_data):
    """
    Generate a simple synthetic ECG signal
    
    Parameters:
    -----------
    patient_data : dict
        Dictionary containing patient clinical data
    
    Returns:
    --------
    ecg_signal : numpy.ndarray
        Synthetic ECG signal
    ecg_time : numpy.ndarray
        Time array for the ECG signal
    """
    print("Generating simple ECG signal...")
    
    # Create a simple sine wave as ECG
    duration = 10  # seconds
    sampling_rate = 100  # Hz
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Base frequency based on heart rate
    heart_rate = patient_data.get('max_heart_rate', 75)
    base_freq = heart_rate / 60  # Convert to Hz
    
    # Create a simple ECG-like signal
    signal = np.zeros_like(t)
    
    # Add QRS complexes
    for i in range(int(duration * base_freq)):
        # Position of the QRS complex
        pos = i / base_freq
        
        # Add a QRS complex (simplified as a spike)
        idx = np.where((t >= pos - 0.1) & (t <= pos + 0.1))[0]
        if len(idx) > 0:
            # Create a spike
            spike = np.sin(np.linspace(0, np.pi, len(idx))) * 0.8
            signal[idx] += spike
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(t))
    signal += noise
    
    print("ECG signal generated!")
    return signal, t

def analyze_simple_ecg(ecg_signal, ecg_time, patient_data):
    """
    Analyze ECG signal for abnormalities (simplified version)
    
    Parameters:
    -----------
    ecg_signal : numpy.ndarray
        ECG signal
    ecg_time : numpy.ndarray
        Time array for the ECG signal
    patient_data : dict
        Dictionary containing patient clinical data
    
    Returns:
    --------
    abnormalities : dict
        Dictionary containing detected abnormalities with timestamps
    """
    print("Analyzing ECG signal...")
    
    # Create some dummy abnormalities
    abnormalities = {
        'PVCs': [],
        'Flatlines': [],
        'Tachycardia': [],
        'Bradycardia': [],
        'QT_prolongation': [],
        'Atrial_Fibrillation': []
    }
    
    # Add some random abnormalities based on patient data
    if patient_data.get('age', 60) > 65:
        # Add a PVC
        abnormalities['PVCs'].append({
            'time': float(np.random.uniform(1, 8)),
            'duration': 0.2
        })
    
    if patient_data.get('max_heart_rate', 75) > 100:
        # Add tachycardia
        abnormalities['Tachycardia'].append({
            'time': float(np.random.uniform(2, 7)),
            'duration': 1.0,
            'rate': float(patient_data.get('max_heart_rate', 75))
        })
    
    if patient_data.get('prior_cardiac_event', {}).get('type', ''):
        # Add QT prolongation
        abnormalities['QT_prolongation'].append({
            'time': float(np.random.uniform(3, 6)),
            'duration': 0.5,
            'interval': 0.48
        })
    
    print("ECG analysis complete!")
    return abnormalities
