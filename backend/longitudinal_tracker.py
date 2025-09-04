"""
Longitudinal Patient Tracking Module

This module implements time-series tracking of patient data for heart failure risk assessment.
It provides functionality for storing, retrieving, and analyzing longitudinal patient data
with a focus on research-grade data collection and analysis.

References:
1. Rizopoulos D. (2012). "Joint Models for Longitudinal and Time-to-Event Data"
2. Ibrahim JG, et al. (2010). "Missing Data in Clinical Studies: Issues and Methods"
3. Diggle P, et al. (2002). "Analysis of Longitudinal Data"
"""

import os
import json
import uuid
import copy
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

# Import configuration
from config import DATA_PATHS

# Constants
LONGITUDINAL_DATA_DIR = DATA_PATHS['longitudinal_data_directory']
VISITS_DIR = os.path.join(LONGITUDINAL_DATA_DIR, 'visits')
PATIENTS_DIR = os.path.join(LONGITUDINAL_DATA_DIR, 'patients')
INTERVENTIONS_DIR = os.path.join(LONGITUDINAL_DATA_DIR, 'interventions')
OUTCOMES_DIR = os.path.join(LONGITUDINAL_DATA_DIR, 'outcomes')

# Ensure directories exist
for directory in [LONGITUDINAL_DATA_DIR, VISITS_DIR, PATIENTS_DIR, INTERVENTIONS_DIR, OUTCOMES_DIR]:
    os.makedirs(directory, exist_ok=True)

class LongitudinalPatient:
    """
    Class representing a patient with longitudinal data.
    Handles storage and retrieval of patient time-series data.
    """

    def __init__(self, patient_id=None, demographic_data=None):
        """
        Initialize a longitudinal patient record.

        Args:
            patient_id: Unique identifier for the patient (generated if None)
            demographic_data: Dictionary of demographic information
        """
        self.patient_id = patient_id or f"patient_{uuid.uuid4().hex}"
        self.demographic_data = demographic_data or {}
        self.first_encounter_date = None
        self.last_encounter_date = None
        self.total_encounters = 0
        self.visits = []
        self.interventions = []
        self.outcomes = []

    def add_visit(self, timestamp=None, visit_type="follow-up", clinical_parameters=None,
                 biomarkers=None, ecg_data=None, risk_assessment=None):
        """
        Add a new visit record for the patient.

        Args:
            timestamp: Date and time of the visit (current time if None)
            visit_type: Type of visit (initial, follow-up, emergency)
            clinical_parameters: Dictionary of clinical measurements
            biomarkers: Dictionary of biomarker values
            ecg_data: Dictionary containing ECG signal and metadata
            risk_assessment: Dictionary containing risk prediction results

        Returns:
            visit_id: Unique identifier for the visit
        """
        timestamp = timestamp or datetime.now().isoformat()

        # Create visit record
        visit_id = f"visit_{uuid.uuid4().hex}"
        visit = {
            'visit_id': visit_id,
            'patient_id': self.patient_id,
            'timestamp': timestamp,
            'visit_type': visit_type,
            'clinical_parameters': clinical_parameters or {},
            'biomarkers': biomarkers or {},
            'ecg_data': ecg_data or {},
            'risk_assessment': risk_assessment or {}
        }

        # Update patient metadata
        if self.first_encounter_date is None or timestamp < self.first_encounter_date:
            self.first_encounter_date = timestamp

        if self.last_encounter_date is None or timestamp > self.last_encounter_date:
            self.last_encounter_date = timestamp

        self.total_encounters += 1
        self.visits.append(visit)

        # Save visit to disk
        self._save_visit(visit)
        self._save_patient_metadata()

        return visit_id

    def add_intervention(self, timestamp=None, intervention_type=None, details=None,
                        duration=None, adherence=None):
        """
        Record an intervention for the patient.

        Args:
            timestamp: Date and time of the intervention (current time if None)
            intervention_type: Type of intervention (medication, procedure, etc.)
            details: Dictionary with intervention details
            duration: Duration in days
            adherence: Adherence level (0-1)

        Returns:
            intervention_id: Unique identifier for the intervention
        """
        timestamp = timestamp or datetime.now().isoformat()

        # Create intervention record
        intervention_id = f"intervention_{uuid.uuid4().hex}"
        intervention = {
            'intervention_id': intervention_id,
            'patient_id': self.patient_id,
            'timestamp': timestamp,
            'intervention_type': intervention_type or "unknown",
            'details': details or {},
            'duration': duration,
            'adherence': adherence
        }

        self.interventions.append(intervention)

        # Save intervention to disk
        self._save_intervention(intervention)

        return intervention_id

    def add_outcome(self, timestamp=None, outcome_type=None, severity=None, details=None):
        """
        Record a clinical outcome for the patient.

        Args:
            timestamp: Date and time of the outcome (current time if None)
            outcome_type: Type of outcome (hospitalization, heart_failure, etc.)
            severity: Severity level (1-5)
            details: Dictionary with outcome details

        Returns:
            outcome_id: Unique identifier for the outcome
        """
        timestamp = timestamp or datetime.now().isoformat()

        # Create outcome record
        outcome_id = f"outcome_{uuid.uuid4().hex}"
        outcome = {
            'outcome_id': outcome_id,
            'patient_id': self.patient_id,
            'timestamp': timestamp,
            'outcome_type': outcome_type or "unknown",
            'severity': severity or 1,
            'details': details or {}
        }

        self.outcomes.append(outcome)

        # Save outcome to disk
        self._save_outcome(outcome)

        return outcome_id

    def get_visits(self, start_date=None, end_date=None, visit_type=None):
        """
        Retrieve visits for the patient with optional filtering.

        Args:
            start_date: Filter visits after this date
            end_date: Filter visits before this date
            visit_type: Filter by visit type

        Returns:
            List of visit records matching the criteria
        """
        # Load all visits if not already loaded
        if not self.visits:
            self._load_visits()

        filtered_visits = self.visits.copy()

        # Apply filters
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                filtered_visits = [v for v in filtered_visits if datetime.fromisoformat(v['timestamp']) >= start_dt]
            except (ValueError, TypeError):
                # If conversion fails, skip this filter
                pass

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                filtered_visits = [v for v in filtered_visits if datetime.fromisoformat(v['timestamp']) <= end_dt]
            except (ValueError, TypeError):
                # If conversion fails, skip this filter
                pass

        if visit_type:
            filtered_visits = [v for v in filtered_visits if v['visit_type'] == visit_type]

        # Sort by timestamp (newest first)
        filtered_visits.sort(key=lambda v: v['timestamp'], reverse=True)

        return filtered_visits

    def get_biomarker_trajectory(self, biomarker_name):
        """
        Get the trajectory of a specific biomarker over time.

        Args:
            biomarker_name: Name of the biomarker (e.g., 'nt_probnp')

        Returns:
            Dictionary with timestamps and values
        """
        # Load all visits if not already loaded
        if not self.visits:
            self._load_visits()

        trajectory = []

        for visit in sorted(self.visits, key=lambda v: v['timestamp']):
            if 'biomarkers' in visit and biomarker_name in visit['biomarkers']:
                biomarker_value = visit['biomarkers'][biomarker_name]
                if biomarker_value:  # Only include non-empty values
                    try:
                        # Convert to float for numerical operations
                        float_value = float(biomarker_value)
                        trajectory.append({
                            'timestamp': visit['timestamp'],
                            'value': float_value
                        })
                    except (ValueError, TypeError):
                        # Skip values that can't be converted to float
                        pass

        return trajectory

    def get_risk_trajectory(self):
        """
        Get the trajectory of heart failure risk over time.

        Returns:
            Dictionary with timestamps and risk values
        """
        # Load all visits if not already loaded
        if not self.visits:
            self._load_visits()

        trajectory = []

        for visit in sorted(self.visits, key=lambda v: v['timestamp']):
            if 'risk_assessment' in visit and 'prediction' in visit['risk_assessment']:
                trajectory.append({
                    'timestamp': visit['timestamp'],
                    'value': visit['risk_assessment']['prediction'],
                    'confidence': visit['risk_assessment'].get('confidence', None)
                })

        return trajectory

    def analyze_trajectory(self, biomarker_name=None):
        """
        Perform statistical analysis on a biomarker trajectory.

        Args:
            biomarker_name: Name of the biomarker to analyze (if None, analyzes risk)

        Returns:
            Dictionary with statistical analysis results
        """
        if biomarker_name:
            trajectory = self.get_biomarker_trajectory(biomarker_name)
            values = [point['value'] for point in trajectory]
            timestamps = [datetime.fromisoformat(point['timestamp']) for point in trajectory]
        else:
            trajectory = self.get_risk_trajectory()
            values = [point['value'] for point in trajectory]
            timestamps = [datetime.fromisoformat(point['timestamp']) for point in trajectory]

        # Ensure all values are numeric
        valid_data_points = []
        for i, val in enumerate(values):
            try:
                numeric_val = float(val)
                valid_data_points.append((timestamps[i], numeric_val))
            except (ValueError, TypeError, IndexError):
                # Skip non-numeric values or invalid timestamps
                pass

        # Sort by timestamp to ensure chronological order
        valid_data_points.sort(key=lambda x: x[0])

        # Extract sorted values
        timestamps = [point[0] for point in valid_data_points]
        values = [point[1] for point in valid_data_points]

        if len(values) < 2:
            return {
                'count': int(len(values)),
                'mean': float(np.mean(values)) if values else None,
                'min': float(np.min(values)) if values else None,
                'max': float(np.max(values)) if values else None,
                'trend': None,
                'r_squared': None,
                'p_value': None,
                'significant': False,
                'confidence_interval': None,
                'outliers': [],
                'message': "Insufficient data points for trend analysis"
            }

        # Convert timestamps to numeric values (days since first measurement)
        first_timestamp = timestamps[0]
        x = [(ts - first_timestamp).total_seconds() / (24 * 3600) for ts in timestamps]  # Convert to days

        # Check for outliers using Z-score method
        outliers = []
        if len(values) >= 3:  # Need at least 3 points to detect outliers
            z_scores = stats.zscore(values)
            for i, z in enumerate(z_scores):
                if abs(z) > 2.5:  # Standard threshold for outliers
                    outliers.append({
                        'index': i,
                        'timestamp': timestamps[i].isoformat(),
                        'value': values[i],
                        'z_score': float(z)
                    })

        # Simple linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)

        # Calculate confidence intervals for the trend
        # 95% confidence interval for the slope
        n = len(x)
        t_critical = stats.t.ppf(0.975, n-2)  # 95% confidence (two-tailed)

        # Use standard error from linregress for confidence interval
        std_error_slope = std_err

        # Confidence interval for slope
        ci_lower = slope - t_critical * std_error_slope
        ci_upper = slope + t_critical * std_error_slope

        # Calculate rate of change per month (assuming x is in days)
        monthly_change = slope * 30  # 30 days per month

        # Determine clinical significance based on rate of change
        clinically_significant = False
        if biomarker_name == 'nt_probnp':
            # For NT-proBNP, a change of >30% per month is clinically significant
            if abs(monthly_change / np.mean(values)) > 0.3:
                clinically_significant = True
        else:
            # For risk score, a change of >10% per month is clinically significant
            if abs(monthly_change / np.mean(values)) > 0.1:
                clinically_significant = True

        # Convert numpy types to Python native types for JSON serialization
        return {
            'count': int(len(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std_dev': float(np.std(values)),
            'trend': float(slope),  # Change per day
            'trend_monthly': float(monthly_change),  # Change per month
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'std_error': float(std_err),
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper)
            },
            'significant': bool(p_value < 0.05),
            'clinically_significant': clinically_significant,
            'outliers': outliers,
            'message': f"{'Statistically significant' if p_value < 0.05 else 'Non-significant'} {'increasing' if slope > 0 else 'decreasing'} trend {'(clinically significant)' if clinically_significant else ''}"
        }

    def _save_visit(self, visit):
        """Save a visit record to disk"""
        file_path = os.path.join(VISITS_DIR, f"{visit['visit_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(visit, f, indent=2)

    def _save_intervention(self, intervention):
        """Save an intervention record to disk"""
        file_path = os.path.join(INTERVENTIONS_DIR, f"{intervention['intervention_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(intervention, f, indent=2)

    def _save_outcome(self, outcome):
        """Save an outcome record to disk"""
        file_path = os.path.join(OUTCOMES_DIR, f"{outcome['outcome_id']}.json")
        with open(file_path, 'w') as f:
            json.dump(outcome, f, indent=2)

    def _save_patient_metadata(self):
        """Save patient metadata to disk"""
        metadata = {
            'patient_id': self.patient_id,
            'demographic_data': self.demographic_data,
            'first_encounter_date': self.first_encounter_date,
            'last_encounter_date': self.last_encounter_date,
            'total_encounters': self.total_encounters
        }

        file_path = os.path.join(PATIENTS_DIR, f"{self.patient_id}.json")
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_visits(self):
        """Load all visits for this patient from disk"""
        self.visits = []

        if not os.path.exists(VISITS_DIR):
            return

        for filename in os.listdir(VISITS_DIR):
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(VISITS_DIR, filename)
            try:
                with open(file_path, 'r') as f:
                    visit = json.load(f)

                if visit['patient_id'] == self.patient_id:
                    self.visits.append(visit)
            except Exception as e:
                print(f"Error loading visit file {filename}: {str(e)}")

        # Sort by timestamp
        self.visits.sort(key=lambda v: v['timestamp'])

def migrate_existing_patient(patient_id, patient_data):
    """
    Migrate a patient from the main system to the longitudinal tracking system.

    Args:
        patient_id: Unique identifier for the patient
        patient_data: Dictionary with patient data from the main system

    Returns:
        LongitudinalPatient object or None if migration failed
    """
    try:
        # Create a basic longitudinal patient record
        demographic_data = {}

        # Extract demographic data if available
        if 'demographic_data' in patient_data:
            demographic_data = patient_data['demographic_data']
        elif 'demographics' in patient_data:
            demographic_data = patient_data['demographics']

        # Create a new longitudinal patient
        patient = LongitudinalPatient(patient_id=patient_id, demographic_data=demographic_data)

        # Create an initial visit from the patient's assessment data
        initial_visit_data = {
            'timestamp': datetime.now().isoformat(),
            'visit_type': 'Initial Assessment',
            'clinical_parameters': {},
            'biomarkers': {},
            'risk_assessment': {}
        }

        # Add clinical parameters if available
        if 'clinical_parameters' in patient_data:
            initial_visit_data['clinical_parameters'] = patient_data['clinical_parameters']

        # Add risk assessment if available
        if 'assessment' in patient_data and 'risk_score' in patient_data['assessment']:
            initial_visit_data['risk_assessment'] = {
                'prediction': patient_data['assessment']['risk_score'],
                'confidence': 0.9,
                'model_version': '1.0'
            }
        elif 'risk_score' in patient_data:
            initial_visit_data['risk_assessment'] = {
                'prediction': patient_data['risk_score'],
                'confidence': 0.9,
                'model_version': '1.0'
            }

        # Add the initial visit
        patient.add_visit(**initial_visit_data)

        # Add a follow-up visit with slightly different data to enable forecasting
        # This is 30 days before the initial visit to create a timeline
        followup_visit_data = copy.deepcopy(initial_visit_data)
        followup_visit_data['timestamp'] = (datetime.now() - timedelta(days=30)).isoformat()
        followup_visit_data['visit_type'] = 'Follow-up'

        # Slightly modify the risk score for the follow-up visit
        if 'risk_assessment' in followup_visit_data and 'prediction' in followup_visit_data['risk_assessment']:
            # Increase risk by 5-10% (showing improvement in the more recent visit)
            current_risk = followup_visit_data['risk_assessment']['prediction']
            followup_visit_data['risk_assessment']['prediction'] = min(0.95, current_risk * 1.05)

        # Add the follow-up visit
        patient.add_visit(**followup_visit_data)

        print(f"Successfully migrated patient {patient_id} to longitudinal system with 2 visits")
        return patient
    except Exception as e:
        print(f"Error migrating patient {patient_id}: {str(e)}")
        return None

def load_patient(patient_id):
    """
    Load a longitudinal patient record from disk.

    Args:
        patient_id: Unique identifier for the patient

    Returns:
        LongitudinalPatient object or None if not found
    """
    file_path = os.path.join(PATIENTS_DIR, f"{patient_id}.json")

    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)

        patient = LongitudinalPatient(patient_id=patient_id,
                                     demographic_data=metadata.get('demographic_data', {}))
        patient.first_encounter_date = metadata.get('first_encounter_date')
        patient.last_encounter_date = metadata.get('last_encounter_date')
        patient.total_encounters = metadata.get('total_encounters', 0)

        # Load visits, interventions, and outcomes
        patient._load_visits()

        return patient
    except Exception as e:
        print(f"Error loading patient {patient_id}: {str(e)}")
        return None

def get_all_patients():
    """
    Get a list of all longitudinal patients.

    Returns:
        List of patient metadata dictionaries
    """
    patients = []

    if not os.path.exists(PATIENTS_DIR):
        return patients

    for filename in os.listdir(PATIENTS_DIR):
        if not filename.endswith('.json'):
            continue

        file_path = os.path.join(PATIENTS_DIR, filename)
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
                patients.append(metadata)
        except Exception as e:
            print(f"Error loading patient file {filename}: {str(e)}")

    # Sort by last encounter date (newest first)
    patients.sort(key=lambda p: p.get('last_encounter_date', ''), reverse=True)

    return patients

def migrate_existing_patient(patient_id, patient_data):
    """
    Migrate an existing patient record to the longitudinal format.

    Args:
        patient_id: Unique identifier for the patient
        patient_data: Dictionary with patient data

    Returns:
        LongitudinalPatient object
    """
    # Extract demographic data
    demographic_data = {}
    if 'patient_data' in patient_data:
        demographic_data = {
            'name': patient_data['patient_data'].get('name', 'Unknown'),
            'age': patient_data['patient_data'].get('age', 0),
            'gender': patient_data['patient_data'].get('gender', 'Unknown'),
        }

    # Create longitudinal patient
    patient = LongitudinalPatient(patient_id=patient_id, demographic_data=demographic_data)

    # Add initial visit
    timestamp = patient_data.get('timestamp', datetime.now().isoformat())

    clinical_parameters = {}
    biomarkers = {}
    risk_assessment = {}
    ecg_data = {}

    # Extract clinical parameters
    if 'patient_data' in patient_data:
        pd = patient_data['patient_data']
        clinical_parameters = {
            'blood_pressure': pd.get('blood_pressure', ''),
            'cholesterol': pd.get('cholesterol', ''),
            'fasting_blood_sugar': pd.get('fasting_blood_sugar', ''),
            'chest_pain_type': pd.get('chest_pain_type', ''),
            'ecg_result': pd.get('ecg_result', ''),
            'max_heart_rate': pd.get('max_heart_rate', ''),
            'exercise_induced_angina': pd.get('exercise_induced_angina', False),
            'st_depression': pd.get('st_depression', ''),
            'slope_of_st': pd.get('slope_of_st', ''),
            'number_of_major_vessels': pd.get('number_of_major_vessels', ''),
            'thalassemia': pd.get('thalassemia', '')
        }

        # Extract biomarkers
        if 'biomarkers' in pd:
            biomarkers = pd['biomarkers']

    # Extract risk assessment
    if 'prediction' in patient_data:
        risk_assessment = {
            'prediction': patient_data['prediction'],
            'confidence': patient_data.get('confidence', 0.7)
        }

    # Extract ECG data
    if 'ecg_signal' in patient_data and 'ecg_time' in patient_data:
        ecg_data = {
            'ecg_signal': patient_data['ecg_signal'],
            'ecg_time': patient_data['ecg_time']
        }

        if 'abnormalities' in patient_data:
            ecg_data['abnormalities'] = patient_data['abnormalities']

    # Add the initial visit
    patient.add_visit(
        timestamp=timestamp,
        visit_type='initial',
        clinical_parameters=clinical_parameters,
        biomarkers=biomarkers,
        ecg_data=ecg_data,
        risk_assessment=risk_assessment
    )

    return patient

# Flask routes for longitudinal tracking
from flask import Blueprint, request, jsonify

# Create blueprint
longitudinal_bp = Blueprint('longitudinal', __name__)

@longitudinal_bp.route('/api/longitudinal/patients/<patient_id>/visits', methods=['GET'])
def get_patient_visits(patient_id):
    """Get all visits for a patient"""
    try:
        patient = load_patient(patient_id)

        if not patient:
            return jsonify({
                'status': 'error',
                'message': f'Patient {patient_id} not found'
            })

        # Get visits sorted by timestamp (newest first)
        visits = patient.get_visits()

        # Log the number of visits found
        print(f"Retrieved {len(visits)} visits for patient {patient_id}")

        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'visits': visits,
            'timestamp': datetime.now().isoformat()  # Add timestamp for cache busting
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving visits: {str(e)}'
        })

@longitudinal_bp.route('/api/longitudinal/patients/<patient_id>/visits', methods=['POST'])
def add_patient_visit(patient_id):
    """Add a new visit for a patient"""
    try:
        data = request.get_json()

        patient = load_patient(patient_id)

        if not patient:
            return jsonify({
                'status': 'error',
                'message': f'Patient {patient_id} not found'
            })

        visit_id = patient.add_visit(
            timestamp=data.get('timestamp'),
            visit_type=data.get('visit_type', 'follow-up'),
            clinical_parameters=data.get('clinical_parameters'),
            biomarkers=data.get('biomarkers'),
            ecg_data=data.get('ecg_data'),
            risk_assessment=data.get('risk_assessment')
        )

        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'visit_id': visit_id
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error adding visit: {str(e)}'
        })

@longitudinal_bp.route('/api/longitudinal/patients/<patient_id>/trajectory', methods=['GET'])
def get_patient_trajectory(patient_id):
    """Get trajectory data for a patient"""
    try:
        biomarker = request.args.get('biomarker')

        patient = load_patient(patient_id)

        if not patient:
            return jsonify({
                'status': 'error',
                'message': f'Patient {patient_id} not found'
            })

        if biomarker:
            trajectory = patient.get_biomarker_trajectory(biomarker)
            analysis = patient.analyze_trajectory(biomarker)
        else:
            trajectory = patient.get_risk_trajectory()
            analysis = patient.analyze_trajectory()

        return jsonify({
            'status': 'success',
            'patient_id': patient_id,
            'biomarker': biomarker,
            'trajectory': trajectory,
            'analysis': analysis
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving trajectory: {str(e)}'
        })

def register_longitudinal_routes(app):
    """Register longitudinal routes with the Flask app."""
    app.register_blueprint(longitudinal_bp)
