"""
Data processing module for the Heart Failure Prediction System
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import json
from datetime import datetime
import config

class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering for the heart failure prediction system
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.categorical_features = [
            'gender', 'chest_pain_type', 'fasting_blood_sugar', 
            'rest_ecg', 'exercise_angina', 'st_slope', 'thalassemia'
        ]
        self.numerical_features = [
            'age', 'resting_bp', 'cholesterol', 'max_heart_rate', 
            'st_depression', 'num_vessels'
        ]
        self.biomarker_features = ['nt_probnp']
        self.ecg_features = [
            'ecg_af_detected', 'ecg_pvc_count', 'ecg_qt_prolongation'
        ]
        self.target = 'heart_failure'
        
    def load_dataset(self, dataset_name):
        """
        Load a dataset by name
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            pandas DataFrame containing the dataset
        """
        if dataset_name not in config.DATA_PATHS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        file_path = config.DATA_PATHS[dataset_name]
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        return pd.read_csv(file_path)
    
    def preprocess_data(self, data, for_training=True):
        """
        Preprocess the data for model input
        
        Args:
            data: pandas DataFrame containing raw data
            for_training: Whether this preprocessing is for training data
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Extract features
        df = self._extract_features(df)
        
        # Scale numerical features
        df = self._scale_features(df, for_training)
        
        # Handle categorical features
        df = self._encode_categorical_features(df)
        
        # Create interaction terms
        df = self._create_interaction_terms(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # For numerical features, use median imputation
        if any(df[self.numerical_features].isna().any()):
            df[self.numerical_features] = self.imputer.fit_transform(df[self.numerical_features])
        
        # For categorical features, use mode imputation
        for col in self.categorical_features:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
        return df
    
    def _extract_features(self, df):
        """Extract and transform features"""
        # Process NT-proBNP with age-adjusted thresholds
        if 'nt_probnp' in df.columns and 'age' in df.columns:
            df['nt_probnp_threshold'] = df['age'].apply(self._get_nt_probnp_threshold)
            df['nt_probnp_ratio'] = df['nt_probnp'] / df['nt_probnp_threshold']
            df['nt_probnp_log'] = np.log1p(df['nt_probnp'])
            df['nt_probnp_normalized'] = 1 / (1 + np.exp(-0.003 * (df['nt_probnp'] - df['nt_probnp_threshold'])))
        
        # Extract blood pressure components if combined
        if 'blood_pressure' in df.columns and 'systolic_bp' not in df.columns:
            df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(float)
        
        # Process prior cardiac events if available
        if 'prior_cardiac_event' in df.columns and 'time_since_event' in df.columns:
            # Apply time-based decay to prior event impact
            df['prior_event_impact'] = df['prior_cardiac_event'] * np.exp(-0.001 * df['time_since_event'])
        
        return df
    
    def _scale_features(self, df, for_training):
        """Scale numerical features"""
        numerical_cols = [col for col in self.numerical_features if col in df.columns]
        
        if not numerical_cols:
            return df
            
        if for_training:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical features"""
        # Gender encoding
        if 'gender' in df.columns:
            df['gender_male'] = (df['gender'] == 'Male').astype(int)
            
        # Chest pain type one-hot encoding
        if 'chest_pain_type' in df.columns:
            pain_types = pd.get_dummies(df['chest_pain_type'], prefix='cp')
            df = pd.concat([df, pain_types], axis=1)
            
        # ST slope one-hot encoding
        if 'st_slope' in df.columns:
            df['st_slope_flat'] = (df['st_slope'] == 'Flat').astype(int)
            df['st_slope_downsloping'] = (df['st_slope'] == 'Downsloping').astype(int)
            
        # Thalassemia one-hot encoding
        if 'thalassemia' in df.columns:
            df['thalassemia_fixed'] = (df['thalassemia'] == 'Fixed Defect').astype(int)
            df['thalassemia_reversible'] = (df['thalassemia'] == 'Reversible Defect').astype(int)
            
        return df
    
    def _create_interaction_terms(self, df):
        """Create interaction terms between features"""
        # Age and blood pressure interaction
        if 'age' in df.columns and 'systolic_bp' in df.columns:
            df['age_systolic_interaction'] = df['age'] * df['systolic_bp']
            
        # Exercise angina and ST depression interaction
        if 'exercise_angina' in df.columns and 'st_depression' in df.columns:
            df['angina_st_interaction'] = df['exercise_angina'] * df['st_depression']
            
        # NT-proBNP interactions
        if 'nt_probnp' in df.columns:
            if 'age' in df.columns:
                df['nt_probnp_age_interaction'] = df['nt_probnp'] * df['age']
            if 'st_depression' in df.columns:
                df['nt_probnp_st_interaction'] = df['nt_probnp'] * df['st_depression']
            if 'num_vessels' in df.columns:
                df['nt_probnp_vessels_interaction'] = df['nt_probnp'] * df['num_vessels']
                
        return df
    
    def _get_nt_probnp_threshold(self, age):
        """Get age-adjusted NT-proBNP threshold"""
        if age < 50:
            return config.NT_PROBNP_THRESHOLDS['age_lt_50']
        elif age <= 75:
            return config.NT_PROBNP_THRESHOLDS['age_50_to_75']
        else:
            return config.NT_PROBNP_THRESHOLDS['age_gt_75']
    
    def prepare_training_data(self, data, test_size=0.2, random_state=42):
        """
        Prepare data for model training
        
        Args:
            data: Raw data DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Preprocess the data
        processed_data = self.preprocess_data(data, for_training=True)
        
        # Separate features and target
        if self.target not in processed_data.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
            
        X = processed_data.drop(self.target, axis=1)
        y = processed_data[self.target]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        return X_train_resampled, X_test, y_train_resampled, y_test
    
    def prepare_patient_data(self, patient_data):
        """
        Prepare a single patient's data for prediction
        
        Args:
            patient_data: Dictionary containing patient data
            
        Returns:
            Processed DataFrame ready for model input
        """
        # Convert dictionary to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess
        processed_df = self.preprocess_data(df, for_training=False)
        
        return processed_df
    
    def save_patient_data(self, patient_id, patient_data):
        """
        Save patient data for longitudinal tracking
        
        Args:
            patient_id: Unique identifier for the patient
            patient_data: Dictionary containing patient data
            
        Returns:
            Success status
        """
        # Add timestamp if not present
        if 'timestamp' not in patient_data:
            patient_data['timestamp'] = datetime.now().isoformat()
            
        # Create patient directory if it doesn't exist
        patient_dir = os.path.join(config.DATA_PATHS['patient_data_directory'], str(patient_id))
        os.makedirs(patient_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.fromisoformat(patient_data['timestamp'])
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(patient_dir, filename)
        
        # Save data as JSON
        with open(file_path, 'w') as f:
            json.dump(patient_data, f, indent=2)
            
        return True
    
    def get_patient_history(self, patient_id):
        """
        Retrieve a patient's historical data
        
        Args:
            patient_id: Unique identifier for the patient
            
        Returns:
            List of patient data records sorted by timestamp
        """
        patient_dir = os.path.join(config.DATA_PATHS['patient_data_directory'], str(patient_id))
        
        if not os.path.exists(patient_dir):
            return []
            
        # Load all JSON files in the patient directory
        records = []
        for filename in os.listdir(patient_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(patient_dir, filename)
                with open(file_path, 'r') as f:
                    record = json.load(f)
                    records.append(record)
                    
        # Sort by timestamp
        records.sort(key=lambda x: x.get('timestamp', ''))
        
        return records
    
    def extract_temporal_features(self, patient_history):
        """
        Extract temporal features from patient history
        
        Args:
            patient_history: List of patient data records
            
        Returns:
            Dictionary of temporal features
        """
        if not patient_history or len(patient_history) < config.LONGITUDINAL_CONFIG['min_data_points']:
            return {}
            
        # Convert timestamps to datetime objects
        for record in patient_history:
            if 'timestamp' in record:
                record['timestamp_dt'] = datetime.fromisoformat(record['timestamp'])
                
        # Sort by timestamp
        patient_history.sort(key=lambda x: x.get('timestamp_dt', datetime.min))
        
        # Extract temporal features
        temporal_features = {}
        
        # Calculate slopes for key parameters
        for param in ['systolic_bp', 'diastolic_bp', 'cholesterol', 'nt_probnp', 'max_heart_rate']:
            values = [record.get(param) for record in patient_history if param in record]
            times = [record.get('timestamp_dt') for record in patient_history if param in record]
            
            if len(values) >= 2:
                # Convert times to days since first measurement
                time_deltas = [(t - times[0]).days for t in times]
                
                # Calculate slope using numpy's polyfit
                if any(td != 0 for td in time_deltas):  # Avoid division by zero
                    slope, _ = np.polyfit(time_deltas, values, 1)
                    temporal_features[f'{param}_slope'] = slope
                    
                    # Calculate volatility (standard deviation)
                    temporal_features[f'{param}_volatility'] = np.std(values)
                    
                    # Calculate recent trend (using last few measurements)
                    window = min(config.LONGITUDINAL_CONFIG['trend_window'], len(values))
                    recent_values = values[-window:]
                    recent_times = time_deltas[-window:]
                    
                    if len(recent_values) >= 2 and any(t != recent_times[0] for t in recent_times):
                        recent_slope, _ = np.polyfit(recent_times, recent_values, 1)
                        temporal_features[f'{param}_recent_trend'] = recent_slope
        
        # Calculate frequency of abnormal events
        if len(patient_history) >= 2:
            total_days = (patient_history[-1]['timestamp_dt'] - patient_history[0]['timestamp_dt']).days
            if total_days > 0:
                # Count ECG abnormalities
                af_count = sum(1 for record in patient_history if record.get('ecg_af_detected', 0) > 0)
                pvc_count = sum(1 for record in patient_history if record.get('ecg_pvc_count', 0) > 0)
                
                temporal_features['af_frequency'] = af_count / (total_days / 365)  # Annual frequency
                temporal_features['pvc_frequency'] = pvc_count / (total_days / 365)  # Annual frequency
                
        return temporal_features
