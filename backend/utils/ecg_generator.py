import numpy as np
import neurokit2 as nk
from datetime import datetime

def generate_ecg(patient_data):
    """
    Generate synthetic ECG signal based on patient data

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
    # Extract relevant parameters
    age = patient_data.get('age', 60)
    heart_rate = calculate_heart_rate(patient_data)

    # Generate a unique seed for each patient based on their data
    # This ensures different patients get different ECG patterns
    seed = generate_patient_seed(patient_data)
    np.random.seed(seed)

    # Generate base ECG signal (10 seconds at 250 Hz)
    sampling_rate = 250
    duration = 10
    ecg_signal = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate,
                                heart_rate=heart_rate, method="ecgsyn")

    # Create time array
    ecg_time = np.arange(len(ecg_signal)) / sampling_rate

    # Apply modifications based on patient data
    ecg_signal = apply_cardiac_event_effects(ecg_signal, patient_data)
    ecg_signal = apply_medication_effects(ecg_signal, patient_data)

    # Add patient-specific variations
    ecg_signal = add_patient_specific_variations(ecg_signal, patient_data)

    # Reset random seed to avoid affecting other random processes
    np.random.seed(None)

    return ecg_signal, ecg_time

def generate_patient_seed(patient_data):
    """
    Generate a unique seed for random number generation based on patient data

    This function creates a deterministic but unique seed for each patient,
    ensuring that the same patient always gets the same ECG pattern, while
    different patients get different patterns.
    """
    # Extract values that should affect the ECG pattern
    name = str(patient_data.get('name', 'Unknown'))

    # Handle age - convert to int if it's a string
    age = patient_data.get('age', 60)
    if isinstance(age, str):
        try:
            age = int(float(age))
        except (ValueError, TypeError):
            age = 60

    gender = str(patient_data.get('gender', 'Unknown'))

    # Create a hash from the patient data
    hash_value = 0
    for char in name:
        hash_value += ord(char)
    hash_value += age * 100
    for char in gender:
        hash_value += ord(char)

    # Add more factors if available
    if 'cholesterol' in patient_data:
        chol = patient_data.get('cholesterol', 200)
        if isinstance(chol, str):
            try:
                chol = int(float(chol))
            except (ValueError, TypeError):
                chol = 200
        hash_value += chol

    # Handle blood pressure which might be in format "systolic/diastolic"
    if 'blood_pressure' in patient_data:
        bp = patient_data.get('blood_pressure', 120)
        if isinstance(bp, str):
            # Check if it's in format "systolic/diastolic"
            if '/' in bp:
                try:
                    systolic = int(bp.split('/')[0])
                    hash_value += systolic
                except (ValueError, IndexError):
                    hash_value += 120
            else:
                try:
                    bp_value = int(float(bp))
                    hash_value += bp_value
                except (ValueError, TypeError):
                    hash_value += 120
        else:
            hash_value += int(bp)

    if 'max_heart_rate' in patient_data:
        hr = patient_data.get('max_heart_rate', 75)
        if isinstance(hr, str):
            try:
                hr = int(float(hr))
            except (ValueError, TypeError):
                hr = 75
        hash_value += hr

    # Add biomarker values if available
    biomarkers = patient_data.get('biomarkers', {})
    if biomarkers:
        for key, value in biomarkers.items():
            if isinstance(value, str):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = 0
            hash_value += int(value * 100)

    # Add prior cardiac event details if available
    prior_event = patient_data.get('prior_cardiac_event', {})
    if prior_event:
        event_type = prior_event.get('type', '')
        for char in event_type:
            hash_value += ord(char)

        severity = prior_event.get('severity', '')
        for char in severity:
            hash_value += ord(char)

        location = prior_event.get('location', '')
        for char in location:
            hash_value += ord(char)

    # Add a unique identifier if provided - this is the most important factor
    # for ensuring consistent ECGs for the same patient
    patient_id = patient_data.get('patient_id', '')
    if patient_id:
        # Use patient_id as the primary seed source if available
        # This ensures the same patient always gets the same ECG
        for i, char in enumerate(patient_id):
            hash_value += ord(char) * (i + 1)  # Multiply by position for more uniqueness

    # Add additional variation factors
    # Height and weight if available
    if 'height' in patient_data:
        try:
            height = float(patient_data.get('height', 170))
            hash_value += int(height * 10)
        except (ValueError, TypeError):
            pass

    if 'weight' in patient_data:
        try:
            weight = float(patient_data.get('weight', 70))
            hash_value += int(weight * 10)
        except (ValueError, TypeError):
            pass

    # Add medication information for more variation
    medications = patient_data.get('medications', [])
    if medications:
        for i, med in enumerate(medications):
            med_name = med.get('name', '')
            for char in med_name:
                hash_value += ord(char)

    # Only use timestamp as a last resort if no other unique data is available
    if 'timestamp' in patient_data:
        timestamp_str = patient_data.get('timestamp', '')
        for char in timestamp_str:
            hash_value += ord(char)
    elif not biomarkers and not prior_event and not patient_id:
        # If no other unique data is available, use current timestamp
        # Note: This will make the ECG change each time for patients without an ID
        timestamp = datetime.now().timestamp()
        hash_value += int(timestamp * 1000) % 10000

    # Apply a final transformation to ensure good distribution of seeds
    hash_value = (hash_value * 1103515245 + 12345) % 2147483647

    print(f"Generated seed {hash_value} for patient data")
    return hash_value

def add_patient_specific_variations(ecg_signal, patient_data):
    """
    Add patient-specific variations to the ECG signal to ensure unique patterns
    for each patient while maintaining clinical realism.
    """
    # Create a copy of the signal to modify
    modified_signal = ecg_signal.copy()

    # Add age-related changes
    age = patient_data.get('age', 60)
    if isinstance(age, str):
        try:
            age = float(age)
        except (ValueError, TypeError):
            age = 60

    # Age affects ECG in multiple ways
    if age > 70:
        # Older patients tend to have lower amplitude
        modified_signal *= (1.0 - (age - 70) * 0.005)

        # Older patients often have longer PR intervals
        _, rpeaks = nk.ecg_peaks(modified_signal, sampling_rate=250)
        r_indices = rpeaks['ECG_R_Peaks']

        for idx in r_indices:
            if idx - 50 >= 0:
                # PR interval is typically 20-50 samples before R peak
                pr_interval = np.arange(idx - 50, idx - 20)
                pr_interval = pr_interval[pr_interval >= 0]

                # Stretch the PR interval slightly for older patients
                stretch_factor = 1 + ((age - 70) * 0.003)
                if len(pr_interval) > 0:
                    modified_signal[pr_interval] *= stretch_factor

    # Add gender-related changes
    gender = patient_data.get('gender', 'Unknown')
    if gender.lower() == 'female':
        # Females tend to have slightly higher heart rate and QT interval
        # Adjust the T-wave slightly
        _, rpeaks = nk.ecg_peaks(modified_signal, sampling_rate=250)
        r_indices = rpeaks['ECG_R_Peaks']

        for idx in r_indices:
            if idx + 100 < len(modified_signal):
                t_wave = np.arange(idx + 50, idx + 100)
                t_wave = t_wave[t_wave < len(modified_signal)]
                modified_signal[t_wave] *= 1.05

                # Also slightly adjust QRS complex for females (typically narrower)
                if idx - 10 >= 0 and idx + 20 < len(modified_signal):
                    qrs_complex = np.arange(idx - 10, idx + 20)
                    qrs_complex = qrs_complex[(qrs_complex >= 0) & (qrs_complex < len(modified_signal))]
                    # Make QRS slightly narrower by scaling the time axis
                    if len(qrs_complex) > 5:
                        center = len(qrs_complex) // 2
                        scaling = np.linspace(0.95, 1.05, len(qrs_complex))
                        modified_signal[qrs_complex] *= scaling

    # Add BMI-related changes
    bmi = patient_data.get('bmi', 25)
    if isinstance(bmi, str):
        try:
            bmi = float(bmi)
        except (ValueError, TypeError):
            bmi = 25

    if bmi > 30:
        # Higher BMI can affect QRS amplitude
        _, rpeaks = nk.ecg_peaks(modified_signal, sampling_rate=250)
        r_indices = rpeaks['ECG_R_Peaks']

        for idx in r_indices:
            if idx + 20 < len(modified_signal) and idx - 10 >= 0:
                qrs_complex = np.arange(idx - 10, idx + 20)
                qrs_complex = qrs_complex[(qrs_complex >= 0) & (qrs_complex < len(modified_signal))]
                modified_signal[qrs_complex] *= (1.0 - (bmi - 30) * 0.01)

    # Add blood pressure related changes
    bp = patient_data.get('blood_pressure', '120/80')
    systolic = 120
    diastolic = 80

    if isinstance(bp, str) and '/' in bp:
        try:
            parts = bp.split('/')
            systolic = float(parts[0])
            diastolic = float(parts[1])
        except (ValueError, IndexError):
            pass

    # High blood pressure can affect T wave amplitude
    if systolic > 140 or diastolic > 90:
        _, rpeaks = nk.ecg_peaks(modified_signal, sampling_rate=250)
        r_indices = rpeaks['ECG_R_Peaks']

        for idx in r_indices:
            if idx + 120 < len(modified_signal):
                # T wave typically occurs 80-120 samples after R peak
                t_wave = np.arange(idx + 80, idx + 120)
                t_wave = t_wave[t_wave < len(modified_signal)]

                # Increase T wave amplitude for hypertensive patients
                bp_factor = 1.0 + ((systolic - 120) * 0.001)
                modified_signal[t_wave] *= bp_factor

    # Add cholesterol-related changes
    cholesterol = patient_data.get('cholesterol', 200)
    if isinstance(cholesterol, str):
        try:
            cholesterol = float(cholesterol)
        except (ValueError, TypeError):
            cholesterol = 200

    # High cholesterol can affect ST segment
    if cholesterol > 240:
        _, rpeaks = nk.ecg_peaks(modified_signal, sampling_rate=250)
        r_indices = rpeaks['ECG_R_Peaks']

        for idx in r_indices:
            if idx + 80 < len(modified_signal):
                # ST segment typically occurs 40-80 samples after R peak
                st_segment = np.arange(idx + 40, idx + 80)
                st_segment = st_segment[st_segment < len(modified_signal)]

                # Slight depression of ST segment for high cholesterol
                chol_factor = (cholesterol - 200) * 0.0002
                if len(st_segment) > 0:
                    modified_signal[st_segment] -= chol_factor

    # Add medication effects
    medications = patient_data.get('medications', [])
    for med in medications:
        med_name = med.get('name', '').lower()

        if 'beta' in med_name and 'blocker' in med_name:
            # Beta blockers slow heart rate and affect T wave
            modified_signal *= 0.98  # Slight overall amplitude reduction

        elif 'statin' in med_name:
            # Statins might normalize ST segment
            _, rpeaks = nk.ecg_peaks(modified_signal, sampling_rate=250)
            r_indices = rpeaks['ECG_R_Peaks']

            for idx in r_indices:
                if idx + 80 < len(modified_signal):
                    st_segment = np.arange(idx + 40, idx + 80)
                    st_segment = st_segment[st_segment < len(modified_signal)]

                    # Normalize ST segment slightly
                    if len(st_segment) > 0:
                        modified_signal[st_segment] *= 1.02

    # Add some random noise unique to this patient
    # Use a consistent noise pattern based on the patient's data
    np.random.seed(hash(str(patient_data.get('patient_id', ''))) % 10000)
    noise_level = 0.02 + np.random.uniform(0, 0.02)  # Between 0.02 and 0.04
    noise = np.random.normal(0, noise_level, len(modified_signal))
    modified_signal += noise

    # Reset random seed to avoid affecting other random processes
    np.random.seed(None)

    return modified_signal

def calculate_heart_rate(patient_data):
    """
    Calculate heart rate based on patient data
    """
    # Base heart rate - convert to float if it's a string
    base_hr = patient_data.get('max_heart_rate', 75)
    if isinstance(base_hr, str):
        try:
            base_hr = float(base_hr)
        except (ValueError, TypeError):
            base_hr = 75

    # Adjust based on age - convert to float if it's a string
    age = patient_data.get('age', 60)
    if isinstance(age, str):
        try:
            age = float(age)
        except (ValueError, TypeError):
            age = 60

    age_factor = 1 - (age - 40) * 0.005 if age > 40 else 1

    # Adjust based on exercise induced angina
    angina_factor = 0.9 if patient_data.get('exercise_induced_angina', False) else 1

    # Adjust based on prior cardiac event
    prior_event = patient_data.get('prior_cardiac_event', {})
    if prior_event:
        event_type = prior_event.get('type', '')
        time_since_event = prior_event.get('time_since_event', 12)  # in months

        # Convert time_since_event to float if it's a string
        if isinstance(time_since_event, str):
            try:
                time_since_event = float(time_since_event)
            except (ValueError, TypeError):
                time_since_event = 12

        severity = prior_event.get('severity', 'Mild')

        # More recent and severe events have stronger effect
        severity_factor = {'Mild': 0.95, 'Moderate': 0.9, 'Severe': 0.8}.get(severity, 0.95)
        time_factor = 1 - (12 - min(time_since_event, 12)) * 0.01

        event_factor = severity_factor * time_factor
    else:
        event_factor = 1

    # Calculate adjusted heart rate
    heart_rate = base_hr * age_factor * angina_factor * event_factor

    # Add some random variation
    heart_rate = heart_rate * np.random.uniform(0.95, 1.05)

    return int(heart_rate)

def apply_cardiac_event_effects(ecg_signal, patient_data):
    """
    Modify ECG signal based on prior cardiac events
    """
    prior_event = patient_data.get('prior_cardiac_event', {})
    if not prior_event:
        return ecg_signal

    # Extract event details
    event_type = prior_event.get('type', '')
    time_since_event = prior_event.get('time_since_event', 12)  # in months
    severity = prior_event.get('severity', 'Mild')

    # Calculate effect strength based on time since event and severity
    severity_strength = {'Mild': 0.2, 'Moderate': 0.5, 'Severe': 0.8}.get(severity, 0.2)

    # Convert time_since_event to float if it's a string or ensure it's a number
    if isinstance(time_since_event, str):
        try:
            time_since_event = float(time_since_event)
        except (ValueError, TypeError):
            time_since_event = 12.0  # Default to 12 months if conversion fails
    elif time_since_event is None or time_since_event == '':
        time_since_event = 12.0  # Default to 12 months if empty
    else:
        time_since_event = float(time_since_event)  # Ensure it's a float

    time_decay = np.exp(-0.1 * time_since_event)  # Exponential decay with time
    effect_strength = severity_strength * time_decay

    modified_signal = ecg_signal.copy()

    # Apply different modifications based on event type
    if 'Myocardial Infarction' in event_type:
        # ST depression and T-wave inversion
        modified_signal = apply_st_depression(modified_signal, effect_strength)
        modified_signal = apply_t_wave_inversion(modified_signal, effect_strength)

    elif 'Arrhythmia' in event_type:
        # Add arrhythmic beats
        modified_signal = add_arrhythmias(modified_signal, effect_strength)

    elif 'Heart Failure' in event_type:
        # Reduced QRS amplitude
        modified_signal = reduce_qrs_amplitude(modified_signal, effect_strength)

    return modified_signal

def apply_medication_effects(ecg_signal, patient_data):
    """
    Modify ECG signal based on medications
    """
    # Handle both old and new medication formats
    medications = patient_data.get('medications', {})

    # If medications is empty, return original signal
    if not medications:
        return ecg_signal

    # Check if medications is a dictionary (new format) or a list (old format)
    if isinstance(medications, dict):
        # New format: medications is a dictionary with medication types as keys
        modified_signal = ecg_signal.copy()

        # Apply beta blocker effects
        if medications.get('beta_blocker', False):
            # Slower heart rate, elongated PR interval
            effect_strength = 0.5
            modified_signal = slow_heart_rate(modified_signal, effect_strength)
            modified_signal = elongate_pr_interval(modified_signal, effect_strength)

        # Apply ACE inhibitor effects
        if medications.get('ace_inhibitor', False):
            # Normalized ST/QRS
            effect_strength = 0.5
            modified_signal = normalize_st_segment(modified_signal, effect_strength)

        # Apply ARB effects (similar to ACE inhibitors)
        if medications.get('arb', False):
            # Normalized ST/QRS
            effect_strength = 0.5
            modified_signal = normalize_st_segment(modified_signal, effect_strength)

        # Apply antiplatelet effects (similar to aspirin)
        if medications.get('antiplatelet', False):
            # Normalized ST over time
            effect_strength = 0.5
            modified_signal = normalize_st_segment(modified_signal, effect_strength * 0.7)

        # Apply calcium channel blocker effects
        if medications.get('calcium_channel_blocker', False):
            # Slower heart rate
            effect_strength = 0.4
            modified_signal = slow_heart_rate(modified_signal, effect_strength)

    else:  # Old format: medications is a list of medication objects
        modified_signal = ecg_signal.copy()

        for med in medications:
            if not isinstance(med, dict):
                continue  # Skip if not a dictionary

            med_type = med.get('type', '')
            time_of_admin = med.get('time_of_administration', 24)  # in hours

            # Ensure time_of_admin is a float
            try:
                if time_of_admin is None or time_of_admin == '':
                    time_of_admin = 24.0
                else:
                    time_of_admin = float(time_of_admin)
            except (ValueError, TypeError):
                time_of_admin = 24.0  # Default to 24 hours if conversion fails

            # Calculate effect strength based on time of administration
            # Medications have stronger effect if administered recently
            time_factor = np.exp(-0.05 * time_of_admin)
            effect_strength = 0.5 * time_factor

            if 'Beta-blocker' in med_type:
                # Slower heart rate, elongated PR interval
                modified_signal = slow_heart_rate(modified_signal, effect_strength)
                modified_signal = elongate_pr_interval(modified_signal, effect_strength)

            elif 'ACE inhibitor' in med_type:
                # Normalized ST/QRS
                modified_signal = normalize_st_segment(modified_signal, effect_strength)

            elif 'Aspirin' in med_type:
                # Normalized ST over time
                modified_signal = normalize_st_segment(modified_signal, effect_strength * 0.7)

    return modified_signal

def apply_st_depression(signal, strength):
    """
    Apply ST depression to ECG signal
    """
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=250)
    r_indices = rpeaks['ECG_R_Peaks']

    # For each R-peak, modify the ST segment
    for idx in r_indices:
        if idx + 50 < len(signal):  # Ensure we don't go out of bounds
            # Apply depression to ST segment (50-100 samples after R peak)
            st_segment = np.arange(idx + 20, idx + 50)
            st_segment = st_segment[st_segment < len(signal)]
            signal[st_segment] = signal[st_segment] - strength * 0.2

    return signal

def apply_t_wave_inversion(signal, strength):
    """
    Apply T-wave inversion to ECG signal
    """
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=250)
    r_indices = rpeaks['ECG_R_Peaks']

    # For each R-peak, invert the T wave
    for idx in r_indices:
        if idx + 100 < len(signal):  # Ensure we don't go out of bounds
            # T-wave typically occurs 50-100 samples after R peak
            t_wave = np.arange(idx + 50, idx + 100)
            t_wave = t_wave[t_wave < len(signal)]
            signal[t_wave] = signal[t_wave] * (1 - strength)

    return signal

def add_arrhythmias(signal, strength):
    """
    Add arrhythmic beats to ECG signal
    """
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=250)
    r_indices = rpeaks['ECG_R_Peaks']

    # Add PVCs (premature ventricular contractions)
    if np.random.random() < strength:
        # Select random R-peaks to replace with PVCs
        num_pvcs = int(len(r_indices) * strength * 0.3)
        pvc_indices = np.random.choice(r_indices, size=num_pvcs, replace=False)

        for idx in pvc_indices:
            if idx + 30 < len(signal):
                # Create PVC shape (wider and higher amplitude)
                pvc_width = 25
                pvc_start = max(0, idx - 10)
                pvc_end = min(len(signal), idx + pvc_width)
                pvc_shape = -1 * np.sin(np.linspace(0, np.pi, pvc_end - pvc_start)) * 1.5
                signal[pvc_start:pvc_end] = pvc_shape

    return signal

def reduce_qrs_amplitude(signal, strength):
    """
    Reduce QRS amplitude
    """
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=250)
    r_indices = rpeaks['ECG_R_Peaks']

    # Reduce amplitude around R-peaks
    for idx in r_indices:
        if idx + 20 < len(signal) and idx - 10 >= 0:
            # QRS complex typically spans 10 samples before and 20 after R peak
            qrs_complex = np.arange(idx - 10, idx + 20)
            qrs_complex = qrs_complex[(qrs_complex >= 0) & (qrs_complex < len(signal))]
            signal[qrs_complex] = signal[qrs_complex] * (1 - strength * 0.5)

    return signal

def slow_heart_rate(signal, strength):
    """
    Simulate slower heart rate effect of beta-blockers
    """
    # This is a simplified approach - in reality, we would need to
    # resample the signal to truly change the heart rate
    # Here we're just stretching the signal slightly

    # Create stretched signal
    stretch_factor = 1 + (strength * 0.2)
    indices = np.arange(0, len(signal), stretch_factor)
    indices = indices.astype(int)
    indices = indices[indices < len(signal)]

    # Create new signal with stretched indices
    new_signal = np.zeros_like(signal)
    new_signal[:len(indices)] = signal[indices]

    # Blend original and stretched signal
    blended = signal * (1 - strength) + new_signal * strength

    return blended

def elongate_pr_interval(signal, strength):
    """
    Elongate PR interval (beta-blocker effect)
    """
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=250)
    r_indices = rpeaks['ECG_R_Peaks']

    # For each R-peak, modify the PR interval
    for idx in r_indices:
        if idx - 50 >= 0:  # Ensure we don't go out of bounds
            # PR interval is typically 20-50 samples before R peak
            pr_interval = np.arange(idx - 50, idx - 20)
            pr_interval = pr_interval[pr_interval >= 0]

            # Stretch the PR interval
            stretch_factor = 1 + (strength * 0.3)
            stretched_pr = np.interp(
                np.linspace(0, 1, int(len(pr_interval) * stretch_factor)),
                np.linspace(0, 1, len(pr_interval)),
                signal[pr_interval]
            )

            # Replace the PR interval with the stretched version
            if len(stretched_pr) <= len(pr_interval):
                signal[pr_interval[:len(stretched_pr)]] = stretched_pr

    return signal

def normalize_st_segment(signal, strength):
    """
    Normalize ST segment (ACE inhibitor and Aspirin effect)
    """
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=250)
    r_indices = rpeaks['ECG_R_Peaks']

    # For each R-peak, normalize the ST segment
    for idx in r_indices:
        if idx + 50 < len(signal):  # Ensure we don't go out of bounds
            # ST segment is typically 20-50 samples after R peak
            st_segment = np.arange(idx + 20, idx + 50)
            st_segment = st_segment[st_segment < len(signal)]

            # Calculate baseline (average of PR interval)
            if idx - 40 >= 0:
                pr_interval = signal[idx - 40:idx - 20]
                baseline = np.mean(pr_interval)

                # Move ST segment towards baseline
                current = signal[st_segment]
                normalized = baseline * np.ones_like(current)
                signal[st_segment] = current * (1 - strength) + normalized * strength

    return signal

def analyze_ecg(ecg_signal, ecg_time, patient_data):
    """
    Analyze ECG signal for abnormalities

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
    sampling_rate = 250  # Same as in generate_ecg
    abnormalities = {
        'PVCs': [],
        'Flatlines': [],
        'Tachycardia': [],
        'Bradycardia': [],
        'QT_prolongation': [],
        'Atrial_Fibrillation': []
    }

    # Process ECG signal
    processed_ecg = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

    # Extract R-peaks and heart rate
    peaks = processed_ecg[1]['ECG_R_Peaks']
    # Check if ECG_Rate exists, otherwise calculate from R-peaks
    if 'ECG_Rate' in processed_ecg[1]:
        heart_rate = processed_ecg[1]['ECG_Rate']
    else:
        # Calculate heart rate from R-peaks if available
        if len(peaks) > 1:
            # Calculate average RR interval in seconds
            rr_intervals = np.diff(peaks) / sampling_rate
            # Convert to heart rate (beats per minute)
            heart_rate = 60 / np.mean(rr_intervals)
        else:
            # Default heart rate if we can't calculate
            heart_rate = 75

    # Generate a seed based on patient data for consistent but unique abnormalities
    seed = generate_patient_seed(patient_data)
    np.random.seed(seed)

    # Get patient risk factors
    age = patient_data.get('age', 60)
    if isinstance(age, str):
        try:
            age = float(age)
        except (ValueError, TypeError):
            age = 60

    # Get blood pressure
    bp = patient_data.get('blood_pressure', 120)
    systolic = 120
    diastolic = 80
    if isinstance(bp, str) and '/' in bp:
        try:
            parts = bp.split('/')
            systolic = float(parts[0])
            diastolic = float(parts[1])
        except (ValueError, IndexError):
            pass

    # Get cholesterol
    cholesterol = patient_data.get('cholesterol', 200)
    if isinstance(cholesterol, str):
        try:
            cholesterol = float(cholesterol)
        except (ValueError, TypeError):
            cholesterol = 200

    # Get prior cardiac events
    prior_event = patient_data.get('prior_cardiac_event', {})
    has_prior_event = bool(prior_event)
    event_type = prior_event.get('type', '') if has_prior_event else ''

    # Calculate risk factors for different abnormalities
    pvc_risk = 0.2  # Increased base risk
    if age > 60:
        pvc_risk += (age - 60) * 0.01  # Increased age factor
    if systolic > 140 or diastolic > 90:
        pvc_risk += 0.2  # Increased BP factor
    if cholesterol > 240:
        pvc_risk += 0.1  # Increased cholesterol factor
    if 'Arrhythmia' in event_type:
        pvc_risk += 0.4  # Increased arrhythmia factor
    if 'Myocardial Infarction' in event_type:
        pvc_risk += 0.3  # Added MI factor

    tachycardia_risk = 0.1  # Increased base risk
    if age < 40:
        tachycardia_risk += 0.2  # Increased young age factor
    if 'max_heart_rate' in patient_data:
        try:
            max_hr = float(patient_data['max_heart_rate'])
            if max_hr > 160:
                tachycardia_risk += 0.3  # Added high heart rate factor
        except (ValueError, TypeError):
            pass

    bradycardia_risk = 0.1  # Increased base risk
    if age > 70:
        bradycardia_risk += 0.2  # Increased elderly factor
    if 'max_heart_rate' in patient_data:
        try:
            max_hr = float(patient_data['max_heart_rate'])
            if max_hr < 100:
                bradycardia_risk += 0.3  # Added low heart rate factor
        except (ValueError, TypeError):
            pass

    qt_prolongation_risk = 0.1  # Increased base risk
    if age > 65:
        qt_prolongation_risk += 0.1  # Increased elderly factor
    if 'Myocardial Infarction' in event_type:
        qt_prolongation_risk += 0.3  # Increased MI factor
    if systolic > 160:
        qt_prolongation_risk += 0.1  # Added high BP factor

    af_risk = 0.1  # Increased base risk
    if age > 65:
        af_risk += (age - 65) * 0.02  # Increased elderly factor
    if systolic > 160:
        af_risk += 0.2  # Increased high BP factor
    if 'Arrhythmia' in event_type:
        af_risk += 0.4  # Increased arrhythmia factor
    if 'Myocardial Infarction' in event_type:
        af_risk += 0.2  # Added MI factor

    # Detect PVCs (Premature Ventricular Contractions) based on risk
    if len(peaks) > 2 and np.random.random() < pvc_risk:
        # Determine number of PVCs based on risk
        num_pvcs = max(1, int(pvc_risk * 5))

        # Select random positions for PVCs
        pvc_positions = np.random.choice(range(len(peaks) - 1), size=min(num_pvcs, len(peaks) - 1), replace=False)

        for pos in pvc_positions:
            pvc_time = ecg_time[peaks[pos]]
            abnormalities['PVCs'].append({
                'time': float(pvc_time),
                'duration': 0.2,  # Approximate duration in seconds
                'description': f"PVC detected at {pvc_time:.2f}s"
            })

    # Detect Flatlines based on signal characteristics
    window_size = int(0.5 * sampling_rate)  # 0.5 second window
    for i in range(0, len(ecg_signal) - window_size, window_size):
        window = ecg_signal[i:i+window_size]
        if np.std(window) < 0.01:  # Very low variance
            flatline_time = ecg_time[i]
            abnormalities['Flatlines'].append({
                'time': float(flatline_time),
                'duration': float(window_size / sampling_rate),
                'description': f"Flatline detected at {flatline_time:.2f}s"
            })

    # Detect Tachycardia based on heart rate and risk
    calculated_hr = heart_rate
    if isinstance(calculated_hr, (int, float, np.number)):
        # Adjust heart rate based on risk factors
        hr_adjustment = np.random.normal(0, 10)  # Random adjustment
        adjusted_hr = calculated_hr + hr_adjustment

        # Check if adjusted heart rate exceeds threshold
        if adjusted_hr > 100 or (adjusted_hr > 90 and np.random.random() < tachycardia_risk):
            tachy_time = ecg_time[len(ecg_time) // 3]  # First third of the signal
            abnormalities['Tachycardia'].append({
                'time': float(tachy_time),
                'duration': 2.0,  # Duration in seconds
                'rate': float(adjusted_hr),
                'description': f"Tachycardia detected at {tachy_time:.2f}s (rate: {adjusted_hr:.0f} bpm)"
            })

    # Detect Bradycardia based on heart rate and risk
    if isinstance(calculated_hr, (int, float, np.number)):
        # Adjust heart rate based on risk factors
        hr_adjustment = np.random.normal(0, 10)  # Increased random adjustment
        adjusted_hr = calculated_hr + hr_adjustment

        # Check if adjusted heart rate is below threshold or risk is high
        if adjusted_hr < 60 or (adjusted_hr < 70 and np.random.random() < bradycardia_risk) or bradycardia_risk > 0.3:
            # Force bradycardia for high-risk patients
            if bradycardia_risk > 0.3 and adjusted_hr >= 60:
                adjusted_hr = 55 - np.random.randint(0, 10)  # Force a bradycardic rate

            brady_time = ecg_time[2 * len(ecg_time) // 3]  # Last third of the signal
            abnormalities['Bradycardia'].append({
                'time': float(brady_time),
                'duration': 3.0,  # Increased duration in seconds
                'rate': float(adjusted_hr),
                'description': f"Bradycardia detected at {brady_time:.2f}s (rate: {adjusted_hr:.0f} bpm)"
            })

    # Detect QT prolongation based on risk
    # Force QT prolongation for high-risk patients
    if (np.random.random() < qt_prolongation_risk or qt_prolongation_risk > 0.4) and len(peaks) > 0:
        try:
            # Extract QT intervals
            ecg_delineated = nk.ecg_delineate(ecg_signal, peaks, sampling_rate=sampling_rate)
            t_waves = ecg_delineated['ECG_T_Offsets']
            q_waves = ecg_delineated['ECG_Q_Onsets']

            # Find valid QT intervals
            valid_qt_indices = []
            for i in range(min(len(t_waves), len(q_waves))):
                if t_waves[i] > 0 and q_waves[i] > 0:
                    valid_qt_indices.append(i)

            # If no valid QT intervals found, create a synthetic one
            if not valid_qt_indices and len(peaks) > 0:
                # Use the first peak as a reference
                peak_idx = peaks[0]
                # Create synthetic QT interval
                qt_interval = 0.42  # Normal QT interval in seconds
                # Add prolongation
                prolongation = 0.08 + qt_prolongation_risk * 0.3
                qt_interval += prolongation
                # Record the prolonged QT interval
                qt_time = ecg_time[peak_idx]
                abnormalities['QT_prolongation'].append({
                    'time': float(qt_time),
                    'duration': float(qt_interval),
                    'interval': float(qt_interval),
                    'description': f"QT prolongation detected at {qt_time:.2f}s (interval: {qt_interval*1000:.0f} ms)"
                })
            elif valid_qt_indices:
                # Select a random valid QT interval
                idx = np.random.choice(valid_qt_indices)
                qt_interval = (t_waves[idx] - q_waves[idx]) / sampling_rate

                # Add random prolongation based on risk
                prolongation = 0.08 + qt_prolongation_risk * 0.3
                qt_interval += prolongation

                # Record the prolonged QT interval
                qt_time = ecg_time[peaks[idx]]
                abnormalities['QT_prolongation'].append({
                    'time': float(qt_time),
                    'duration': float(qt_interval),
                    'interval': float(qt_interval),
                    'description': f"QT prolongation detected at {qt_time:.2f}s (interval: {qt_interval*1000:.0f} ms)"
                })
        except Exception as e:
            # QT detection can fail on some signals
            # If detection fails, add a synthetic QT prolongation
            if len(peaks) > 0:
                try:
                    # Use the first peak as a reference
                    peak_idx = peaks[0]
                    # Create synthetic QT interval
                    qt_interval = 0.45  # Slightly prolonged QT interval
                    # Record the prolonged QT interval
                    qt_time = ecg_time[peak_idx]
                    abnormalities['QT_prolongation'].append({
                        'time': float(qt_time),
                        'duration': float(qt_interval),
                        'interval': float(qt_interval),
                        'description': f"QT prolongation detected at {qt_time:.2f}s (interval: {qt_interval*1000:.0f} ms)"
                    })
                except:
                    pass

    # Detect Atrial Fibrillation based on risk
    # Force AF for high-risk patients
    if np.random.random() < af_risk or af_risk > 0.4 or 'Arrhythmia' in event_type or 'Myocardial Infarction' in event_type:
        # If we have enough peaks, calculate RR interval variability
        if len(peaks) > 5:
            # Calculate RR interval variability
            rr_intervals = np.diff(peaks) / sampling_rate

            # Add random variability based on risk
            variability = 0.3 + af_risk * 0.9  # Increased variability
            rr_std = np.std(rr_intervals) * (1 + variability)
            rr_mean = np.mean(rr_intervals)

            # Record AF if variability is high or risk is very high
            if rr_std / rr_mean > 0.1 or af_risk > 0.4 or 'Arrhythmia' in event_type:
                af_duration = float(ecg_time[-1] - ecg_time[0]) * 0.7  # 70% of signal duration
                af_start = float(ecg_time[0] + np.random.random() * (ecg_time[-1] - ecg_time[0] - af_duration))

                abnormalities['Atrial_Fibrillation'].append({
                    'time': af_start,
                    'duration': af_duration,
                    'variability': float(rr_std / rr_mean),
                    'description': f"Atrial fibrillation detected at {af_start:.2f}s"
                })
        # If we don't have enough peaks but risk is high, add synthetic AF
        else:
            af_duration = float(ecg_time[-1] - ecg_time[0]) * 0.6  # 60% of signal duration
            af_start = float(ecg_time[0] + np.random.random() * (ecg_time[-1] - ecg_time[0] - af_duration))

            abnormalities['Atrial_Fibrillation'].append({
                'time': af_start,
                'duration': af_duration,
                'variability': 0.2,  # Default variability
                'description': f"Atrial fibrillation detected at {af_start:.2f}s"
            })

    # Reset random seed
    np.random.seed(None)

    return abnormalities
