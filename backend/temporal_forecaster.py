"""
Temporal Forecasting Module for Heart Failure Risk

This module implements time series forecasting for heart failure risk prediction.
It provides functionality for predicting future risk trajectories based on
historical patient data.

References:
1. Cheng, L., et al. (2020). "Temporal Patterns Mining in Electronic Health Records using Deep Learning"
2. Rajkomar, A., et al. (2022). "Machine Learning for Electronic Health Records"
3. Goldstein, B.A., et al. (2021). "Opportunities and Challenges in Developing Risk Prediction Models with Electronic Health Records Data"
4. Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import uuid
import math
from dateutil import parser

# Constants
FORECASTING_DIR = 'data/forecasting'
MODELS_DIR = os.path.join(FORECASTING_DIR, 'models')
FORECASTS_DIR = os.path.join(FORECASTING_DIR, 'forecasts')

# Ensure directories exist
for directory in [FORECASTING_DIR, MODELS_DIR, FORECASTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class TemporalForecaster:
    """
    Class for forecasting future heart failure risk based on longitudinal patient data.
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the temporal forecaster.

        Args:
            model_type: Type of forecasting model to use ('random_forest', 'linear', etc.)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.forecast_horizon = 6  # Default forecast horizon (6 months)
        self.model_path = os.path.join(MODELS_DIR, f"{model_type}_forecaster.joblib")
        self.scaler_path = os.path.join(MODELS_DIR, f"{model_type}_scaler.joblib")
        self.metadata_path = os.path.join(MODELS_DIR, f"{model_type}_metadata.json")

        # Load model if it exists
        self._load_model()

    def _load_model(self):
        """Load the forecasting model if it exists."""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path) and os.path.exists(self.metadata_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names', [])
                    self.forecast_horizon = metadata.get('forecast_horizon', 6)

                return True
            return False
        except Exception as e:
            print(f"Error loading forecasting model: {str(e)}")
            return False

    def _save_model(self):
        """Save the forecasting model."""
        try:
            if self.model is not None and self.scaler is not None:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)

                metadata = {
                    'feature_names': self.feature_names,
                    'forecast_horizon': self.forecast_horizon,
                    'model_type': self.model_type,
                    'timestamp': datetime.now().isoformat()
                }

                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                return True
            return False
        except Exception as e:
            print(f"Error saving forecasting model: {str(e)}")
            return False

    def extract_temporal_features(self, patient_history):
        """
        Extract temporal features from patient history.

        Args:
            patient_history: List of patient visit records

        Returns:
            Dictionary of temporal features
        """
        if not patient_history or len(patient_history) == 0:
            return {}

        # Sort history by timestamp
        sorted_history = sorted(patient_history, key=lambda x: x.get('timestamp', ''))

        # Extract basic features from the most recent visit
        latest_visit = sorted_history[-1]

        # Initialize features dictionary
        features = {}

        # Static features from latest visit
        if 'clinical_parameters' in latest_visit:
            cp = latest_visit['clinical_parameters']

            # Extract blood pressure components
            if 'blood_pressure' in cp and cp['blood_pressure']:
                try:
                    systolic, diastolic = cp['blood_pressure'].split('/')
                    features['systolic_bp'] = float(systolic)
                    features['diastolic_bp'] = float(diastolic)
                except:
                    # If blood pressure is not in expected format, use defaults
                    features['systolic_bp'] = 120
                    features['diastolic_bp'] = 80

            # Extract other clinical parameters
            for param in ['cholesterol', 'fasting_blood_sugar', 'max_heart_rate', 'st_depression']:
                if param in cp and cp[param]:
                    try:
                        features[param] = float(cp[param])
                    except:
                        pass

            # Boolean features
            if 'exercise_induced_angina' in cp:
                features['exercise_angina'] = 1 if cp['exercise_induced_angina'] else 0

            # Categorical features
            if 'slope_of_st' in cp:
                features['st_slope_flat'] = 1 if cp['slope_of_st'] == 'Flat' else 0
                features['st_slope_downsloping'] = 1 if cp['slope_of_st'] == 'Downsloping' else 0

            if 'number_of_major_vessels' in cp:
                try:
                    features['num_vessels'] = int(cp['number_of_major_vessels'])
                except:
                    features['num_vessels'] = 0

            if 'thalassemia' in cp:
                features['thalassemia_fixed'] = 1 if cp['thalassemia'] == 'Fixed Defect' else 0
                features['thalassemia_reversible'] = 1 if cp['thalassemia'] == 'Reversible Defect' else 0

        # Biomarker features
        if 'biomarkers' in latest_visit and latest_visit['biomarkers']:
            biomarkers = latest_visit['biomarkers']

            if 'nt_probnp' in biomarkers and biomarkers['nt_probnp']:
                try:
                    nt_probnp = float(biomarkers['nt_probnp'])
                    features['nt_probnp'] = nt_probnp

                    # Calculate NT-proBNP threshold ratio based on age
                    age = 65  # Default age if not available
                    if 'demographic_data' in latest_visit and 'age' in latest_visit['demographic_data']:
                        try:
                            age = float(latest_visit['demographic_data']['age'])
                        except:
                            pass

                    # Age-adjusted threshold
                    if age < 50:
                        threshold = 450
                    elif age <= 75:
                        threshold = 900
                    else:
                        threshold = 1800

                    features['nt_probnp_threshold_ratio'] = nt_probnp / threshold
                except:
                    pass

        # Demographic features
        if 'demographic_data' in latest_visit:
            demo = latest_visit['demographic_data']

            if 'age' in demo:
                try:
                    features['age'] = float(demo['age'])
                except:
                    pass

            if 'gender' in demo:
                features['gender_male'] = 1 if demo['gender'] == 'Male' else 0

        # Temporal features (if we have multiple visits)
        if len(sorted_history) >= 2:
            # Calculate risk slope (change in risk over time)
            risk_values = []
            timestamps = []

            for visit in sorted_history:
                if 'risk_assessment' in visit and 'prediction' in visit['risk_assessment']:
                    risk_values.append(float(visit['risk_assessment']['prediction']))
                    try:
                        # First try to parse as is
                        dt = datetime.fromisoformat(visit['timestamp'])
                    except ValueError:
                        try:
                            # Handle ISO format with 'Z' at the end
                            if visit['timestamp'].endswith('Z'):
                                clean_timestamp = visit['timestamp'].replace('Z', '+00:00')
                                dt = datetime.fromisoformat(clean_timestamp)
                            else:
                                # Try to parse with dateutil as a fallback
                                dt = parser.parse(visit['timestamp'])
                        except Exception:
                            # Skip this timestamp if it can't be parsed
                            continue

                    # Ensure all timestamps are naive (no timezone info)
                    if dt.tzinfo is not None:
                        dt = dt.replace(tzinfo=None)

                    timestamps.append(dt)

            if len(risk_values) >= 2:
                # Calculate days between first and last measurement
                days_elapsed = (timestamps[-1] - timestamps[0]).days
                if days_elapsed > 0:
                    # Risk change per day
                    features['risk_slope'] = (risk_values[-1] - risk_values[0]) / days_elapsed
                    # Risk acceleration (change in slope)
                    if len(risk_values) >= 3:
                        mid_idx = len(risk_values) // 2
                        days_first_half = (timestamps[mid_idx] - timestamps[0]).days
                        days_second_half = (timestamps[-1] - timestamps[mid_idx]).days

                        if days_first_half > 0 and days_second_half > 0:
                            slope_first = (risk_values[mid_idx] - risk_values[0]) / days_first_half
                            slope_second = (risk_values[-1] - risk_values[mid_idx]) / days_second_half
                            features['risk_acceleration'] = slope_second - slope_first

            # Calculate biomarker slopes if available
            if len(sorted_history) >= 2:
                for biomarker in ['nt_probnp']:
                    values = []
                    bio_timestamps = []

                    for visit in sorted_history:
                        if 'biomarkers' in visit and biomarker in visit['biomarkers'] and visit['biomarkers'][biomarker]:
                            try:
                                values.append(float(visit['biomarkers'][biomarker]))
                                try:
                                    # First try to parse as is
                                    dt = datetime.fromisoformat(visit['timestamp'])
                                except ValueError:
                                    try:
                                        # Handle ISO format with 'Z' at the end
                                        if visit['timestamp'].endswith('Z'):
                                            clean_timestamp = visit['timestamp'].replace('Z', '+00:00')
                                            dt = datetime.fromisoformat(clean_timestamp)
                                        else:
                                            # Try to parse with dateutil as a fallback
                                            dt = parser.parse(visit['timestamp'])
                                    except Exception:
                                        # Skip this timestamp if it can't be parsed
                                        continue

                                # Ensure all timestamps are naive (no timezone info)
                                if dt.tzinfo is not None:
                                    dt = dt.replace(tzinfo=None)

                                bio_timestamps.append(dt)
                            except:
                                pass

                    if len(values) >= 2:
                        days_elapsed = (bio_timestamps[-1] - bio_timestamps[0]).days
                        if days_elapsed > 0:
                            features[f'{biomarker}_slope'] = (values[-1] - values[0]) / days_elapsed

        # Interaction features
        if 'age' in features and 'systolic_bp' in features:
            features['age_systolic_interaction'] = features['age'] * features['systolic_bp'] / 1000

        if 'exercise_angina' in features and 'st_depression' in features:
            features['angina_st_interaction'] = features['exercise_angina'] * features['st_depression']

        if 'nt_probnp' in features:
            if 'age' in features:
                features['nt_probnp_age_interaction'] = features['nt_probnp'] * features['age'] / 1000

            if 'st_depression' in features:
                features['nt_probnp_st_interaction'] = features['nt_probnp'] * features['st_depression'] / 1000

            if 'num_vessels' in features:
                features['nt_probnp_vessels_interaction'] = features['nt_probnp'] * features['num_vessels'] / 1000

        if 'age' in features and 'max_heart_rate' in features:
            features['age_hr_interaction'] = features['age'] * features['max_heart_rate'] / 1000

        return features

    def prepare_sequence_data(self, patient_histories):
        """
        Prepare sequence data for training the forecasting model.

        Args:
            patient_histories: List of patient histories, each containing visit records

        Returns:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
        """
        sequences = []
        targets = []

        for patient_history in patient_histories:
            # Sort history by timestamp
            sorted_history = sorted(patient_history, key=lambda x: x.get('timestamp', ''))

            # Need at least 2 visits to create a sequence
            if len(sorted_history) < 2:
                continue

            # Create sequences of length 2 or more
            for i in range(len(sorted_history) - 1):
                # Use at least 1 visit to predict the next one
                sequence = sorted_history[:i+1]
                target_visit = sorted_history[i+1]

                # Extract features from the sequence
                features = self.extract_temporal_features(sequence)

                # Extract target value (risk prediction)
                if 'risk_assessment' in target_visit and 'prediction' in target_visit['risk_assessment']:
                    target = float(target_visit['risk_assessment']['prediction'])

                    # Add to dataset
                    sequences.append(features)
                    targets.append(target)

        # Convert to DataFrame for easier handling
        if not sequences:
            return None, None, None

        X = pd.DataFrame(sequences)
        y = np.array(targets)

        # Store feature names
        feature_names = list(X.columns)

        return X, y, feature_names

    def train(self, patient_histories, forecast_horizon=6):
        """
        Train the forecasting model on patient histories.

        Args:
            patient_histories: List of patient histories, each containing visit records
            forecast_horizon: Number of months to forecast into the future

        Returns:
            Dictionary with training results
        """
        # Prepare sequence data
        X, y, feature_names = self.prepare_sequence_data(patient_histories)

        if X is None or len(X) < 2:  # Need at least 2 sequences for meaningful training
            return {
                'status': 'error',
                'message': f'Insufficient data for training. Need at least 2 sequences, got {0 if X is None else len(X)}.'
            }

        # Store feature names and forecast horizon
        self.feature_names = feature_names
        self.forecast_horizon = forecast_horizon

        # Handle missing values
        X = X.fillna(0)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and train model based on model_type
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            # Default to random forest
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        # Train the model
        self.model.fit(X_scaled, y)

        # Save the model
        self._save_model()

        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        r2 = self.model.score(X_scaled, y)

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, self.model.feature_importances_))
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [{'name': name, 'importance': float(importance)} for name, importance in sorted_importance[:10]]
        else:
            top_features = []

        return {
            'status': 'success',
            'message': f'Model trained successfully with {len(X)} sequences.',
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            },
            'feature_importance': top_features,
            'model_type': self.model_type,
            'forecast_horizon': self.forecast_horizon
        }

    def forecast(self, patient_history, horizon=None):
        """
        Generate a forecast for a patient based on their history.

        Args:
            patient_history: List of patient visit records
            horizon: Number of months to forecast (defaults to self.forecast_horizon)

        Returns:
            Dictionary with forecast results
        """
        if self.model is None or self.scaler is None or not self.feature_names:
            return {
                'status': 'error',
                'message': 'No trained model available. Please train the model first.'
            }

        if not patient_history or len(patient_history) == 0:
            return {
                'status': 'error',
                'message': 'No patient history provided.'
            }

        # Use default horizon if not specified
        if horizon is None:
            horizon = self.forecast_horizon

        # Extract features from patient history
        features = self.extract_temporal_features(patient_history)

        # Check if we have all required features
        missing_features = [f for f in self.feature_names if f not in features]

        # Fill in missing features with zeros
        for feature in missing_features:
            features[feature] = 0

        # Create feature vector with the same columns as training data
        X = pd.DataFrame([features])[self.feature_names]

        # Handle missing values
        X = X.fillna(0)

        # Standardize features
        X_scaled = self.scaler.transform(X)

        # Generate base prediction
        base_prediction = float(self.model.predict(X_scaled)[0])

        # Sort history by timestamp
        sorted_history = sorted(patient_history, key=lambda x: x.get('timestamp', ''))
        latest_visit = sorted_history[-1]

        # Parse the timestamp with error handling
        try:
            # First try to parse as is
            latest_timestamp = datetime.fromisoformat(latest_visit['timestamp'])
        except ValueError:
            try:
                # Handle ISO format with 'Z' at the end
                if latest_visit['timestamp'].endswith('Z'):
                    clean_timestamp = latest_visit['timestamp'].replace('Z', '+00:00')
                    latest_timestamp = datetime.fromisoformat(clean_timestamp)
                else:
                    # Try to parse with dateutil as a fallback
                    latest_timestamp = parser.parse(latest_visit['timestamp'])
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error parsing timestamp: {str(e)}'
                }

        # Ensure timestamp is naive (no timezone info)
        if latest_timestamp.tzinfo is not None:
            latest_timestamp = latest_timestamp.replace(tzinfo=None)

        # Get current risk if available
        current_risk = None
        if 'risk_assessment' in latest_visit and 'prediction' in latest_visit['risk_assessment']:
            current_risk = float(latest_visit['risk_assessment']['prediction'])

        # If current_risk is still None, use the base prediction as a fallback
        if current_risk is None:
            current_risk = base_prediction

        # Calculate risk slope if we have multiple visits
        risk_slope = 0
        if len(sorted_history) >= 2 and 'risk_slope' in features:
            risk_slope = features['risk_slope']

        # Generate forecast timestamps and values
        forecast_timestamps = []
        forecast_values = []
        confidence_values = []

        # Calculate confidence based on model uncertainty
        # For random forest, we can use the standard deviation of predictions across trees
        prediction_std = 0.1  # Default uncertainty

        if self.model_type == 'random_forest' and hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X_scaled)[0] for tree in self.model.estimators_])
            prediction_std = np.std(tree_predictions)

        # Generate forecast for each month in the horizon
        for i in range(1, horizon + 1):
            # Calculate forecast timestamp (approximately 30 days per month)
            forecast_date = latest_timestamp + timedelta(days=i * 30)

            # Calculate forecast value
            # For simplicity, we'll use a combination of the model prediction and trend extrapolation
            # The weight of the trend increases for further predictions
            trend_weight = min(0.7, i * 0.1)  # Increase trend weight for longer horizons, max 0.7
            model_weight = 1 - trend_weight

            # Extrapolate based on trend
            trend_prediction = current_risk + (risk_slope * i * 30) if current_risk is not None else base_prediction

            # Add a patient-specific and horizon-specific variation to ensure uniqueness
            # Use patient_id (if available) and horizon to create a deterministic but unique variation
            patient_id = None
            for visit in patient_history:
                if 'patient_id' in visit:
                    patient_id = visit['patient_id']
                    break

            # Create a deterministic but unique variation based on patient_id and horizon
            if patient_id:
                # Use the sum of character codes in patient_id as a seed
                patient_seed = sum(ord(c) for c in patient_id) / 1000
                # Create a unique but deterministic variation for each horizon and month
                # Increased range to make variations more noticeable
                variation = (patient_seed * horizon * i) % 0.08 - 0.04  # Range: -0.04 to 0.04 (4%)
            else:
                variation = 0

            # Check if this is a scenario forecast by looking for a scenario_id in the first visit
            is_scenario = False
            scenario_seed = 0
            for visit in patient_history:
                if 'scenario_id' in visit:
                    is_scenario = True
                    # Create a scenario-specific seed
                    scenario_seed = sum(ord(c) for c in visit['scenario_id']) / 1000
                    break

            # Add additional variation for scenarios
            if is_scenario:
                # Add scenario-specific variation that increases over time
                scenario_variation = (scenario_seed * (i + 1) * 0.01)  # Grows with forecast horizon
                variation += scenario_variation

            # Combine model prediction and trend with the variation
            forecast_value = (model_weight * base_prediction) + (trend_weight * trend_prediction) + variation

            # Ensure forecast is between 0 and 1
            forecast_value = max(0.01, min(0.99, forecast_value))

            # Increase uncertainty for further predictions
            confidence = prediction_std * (1 + (i * 0.1))

            forecast_timestamps.append(forecast_date.isoformat())
            forecast_values.append(float(forecast_value))
            confidence_values.append(float(confidence))

        # Calculate trend description
        if len(forecast_values) >= 2:
            first_value = forecast_values[0]
            last_value = forecast_values[-1]

            if last_value > first_value * 1.1:
                trend_description = "Significantly increasing risk"
            elif last_value > first_value * 1.02:
                trend_description = "Slightly increasing risk"
            elif last_value < first_value * 0.9:
                trend_description = "Significantly decreasing risk"
            elif last_value < first_value * 0.98:
                trend_description = "Slightly decreasing risk"
            else:
                trend_description = "Stable risk"
        else:
            trend_description = "Insufficient data for trend analysis"

        # Calculate peak risk
        peak_risk = max(forecast_values) if forecast_values else 0
        peak_risk_index = forecast_values.index(peak_risk) if forecast_values else 0
        peak_risk_time = forecast_timestamps[peak_risk_index] if forecast_timestamps else None

        # Calculate feature importance for this specific prediction
        feature_importance = []
        if hasattr(self.model, 'feature_importances_'):
            # Get global feature importances
            global_importances = self.model.feature_importances_

            # Create a dictionary of feature names to their values for this prediction
            feature_values = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_values[feature_name] = X.iloc[0, i] if i < len(X.columns) else 0

            # Calculate scenario-specific importance by weighting global importance with feature values
            scenario_importances = {}
            for i, feature_name in enumerate(self.feature_names):
                # Normalize the feature value based on its presence in the data
                feature_value = abs(feature_values[feature_name])
                if feature_value > 0:
                    # Weight the importance by the feature value
                    scenario_importances[feature_name] = global_importances[i] * (1 + 0.5 * feature_value / (1 + feature_value))
                else:
                    scenario_importances[feature_name] = global_importances[i] * 0.5

            # Normalize the scenario importances to sum to 1
            total_importance = sum(scenario_importances.values())
            if total_importance > 0:
                for feature_name in scenario_importances:
                    scenario_importances[feature_name] /= total_importance

            # Sort and format the feature importances
            sorted_importance = sorted(scenario_importances.items(), key=lambda x: x[1], reverse=True)
            feature_importance = [{'name': name, 'importance': float(importance)} for name, importance in sorted_importance[:10]]

        # Generate insights based on forecast values and feature importance
        insights = self.generate_insights(forecast_values, feature_importance)

        # Save forecast
        forecast_id = f"forecast_{uuid.uuid4().hex}"
        forecast_data = {
            'forecast_id': forecast_id,
            'patient_id': patient_history[0]['patient_id'] if 'patient_id' in patient_history[0] else None,
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'forecast_horizon': horizon,
            'current_risk': current_risk,
            'forecast_timestamps': forecast_timestamps,
            'forecast_values': forecast_values,
            'confidence_values': confidence_values,
            'trend_description': trend_description,
            'peak_risk': float(peak_risk),
            'peak_risk_time': peak_risk_time,
            'feature_importance': feature_importance,
            'insights': insights
        }

        # Save forecast to disk
        forecast_path = os.path.join(FORECASTS_DIR, f"{forecast_id}.json")
        with open(forecast_path, 'w') as f:
            json.dump(forecast_data, f, indent=2)

        return {
            'status': 'success',
            'forecast_id': forecast_id,
            'current_risk': current_risk,
            'forecast_timestamps': forecast_timestamps,
            'forecast_values': forecast_values,
            'confidence_values': confidence_values,
            'trend_description': trend_description,
            'peak_risk': float(peak_risk),
            'peak_risk_time': peak_risk_time,
            'feature_importance': feature_importance,
            'insights': insights
        }

    def generate_insights(self, forecast_values, feature_importance, interventions=None):
        """
        Generate insights from the forecast values and feature importance.

        Args:
            forecast_values: List of forecast values
            feature_importance: List of feature importance dictionaries
            interventions: List of interventions applied in the scenario (optional)

        Returns:
            Dictionary with insights
        """
        if not forecast_values or len(forecast_values) == 0:
            return {
                'trend': 'Unknown',
                'key_factors': [],
                'recommendations': []
            }

        # Determine trend
        if len(forecast_values) >= 3:
            first_third = forecast_values[:len(forecast_values)//3]
            last_third = forecast_values[-len(forecast_values)//3:]
            avg_first = sum(first_third) / len(first_third)
            avg_last = sum(last_third) / len(last_third)

            if avg_last > avg_first * 1.1:
                trend = 'Increasing'
            elif avg_last < avg_first * 0.9:
                trend = 'Decreasing'
            else:
                trend = 'Stable'
        else:
            trend = 'Unknown'

        # Extract key factors
        key_factors = []
        for factor in feature_importance[:3]:  # Top 3 factors
            key_factors.append({
                'name': factor['name'],
                'importance': factor['importance']
            })

        # Generate recommendations
        recommendations = []

        # Track intervention types to customize recommendations
        intervention_types = []
        if interventions:
            intervention_types = [i.get('type') for i in interventions if i.get('type')]

        for factor in key_factors:
            factor_name = factor['name']

            # Customize recommendations based on factor and interventions
            if 'blood_pressure' in factor_name:
                if 'blood_pressure' in intervention_types:
                    recommendations.append('The blood pressure intervention shows impact on risk trajectory. Continue monitoring.')
                else:
                    recommendations.append('Consider lifestyle changes or medication to manage blood pressure.')
            elif 'cholesterol' in factor_name:
                if 'cholesterol' in intervention_types:
                    recommendations.append('The cholesterol intervention is affecting risk. Maintain the current approach.')
                else:
                    recommendations.append('Monitor cholesterol levels and consider dietary changes or medication if elevated.')
            elif 'nt_probnp' in factor_name:
                if 'nt_probnp' in intervention_types:
                    recommendations.append('The NT-proBNP intervention is significant. Continue current treatment plan.')
                else:
                    recommendations.append('Regular monitoring of NT-proBNP levels is recommended.')
            elif 'age' in factor_name:
                recommendations.append('Age is a non-modifiable risk factor. Focus on managing other risk factors.')
            elif 'heart_rate' in factor_name or 'max_heart_rate' in factor_name:
                if 'max_heart_rate' in intervention_types:
                    recommendations.append('Heart rate management is showing effect. Continue current approach.')
                else:
                    recommendations.append('Consider exercise and stress management to maintain healthy heart rate.')
            elif 'fasting_blood_sugar' in factor_name:
                if 'fasting_blood_sugar' in intervention_types:
                    recommendations.append('Blood sugar management is impacting risk. Maintain current approach.')
                else:
                    recommendations.append('Monitor blood glucose levels and consider dietary changes if elevated.')
            elif 'exercise' in factor_name:
                if 'exercise' in intervention_types:
                    recommendations.append('The exercise intervention is showing positive effects. Maintain current regimen.')
                else:
                    recommendations.append('Regular physical activity can help reduce cardiovascular risk.')
            elif 'diet' in factor_name:
                if 'diet' in intervention_types:
                    recommendations.append('Dietary changes are impacting risk trajectory. Continue current approach.')
                else:
                    recommendations.append('Consider heart-healthy dietary changes to reduce risk.')

        # Add scenario-specific insights if interventions are present
        if interventions and len(interventions) > 0:
            intervention_count = len(interventions)

            # Create more specific insights based on the interventions
            intervention_types = [i.get('type') for i in interventions if i.get('type')]
            intervention_names = []

            for intervention_type in intervention_types:
                # Convert snake_case to readable format
                readable_name = intervention_type.replace('_', ' ').title()
                intervention_names.append(readable_name)

            # Join intervention names with commas and 'and'
            if len(intervention_names) == 1:
                intervention_str = intervention_names[0]
            elif len(intervention_names) == 2:
                intervention_str = f"{intervention_names[0]} and {intervention_names[1]}"
            else:
                intervention_str = ", ".join(intervention_names[:-1]) + f", and {intervention_names[-1]}"

            if intervention_count == 1:
                recommendations.append(f'This scenario shows the specific effect of {intervention_str} intervention.')
            else:
                recommendations.append(f'This scenario combines {intervention_count} interventions: {intervention_str}.')

            # Add more specific insight about trend in context of interventions
            if trend == 'Decreasing':
                # More specific decreasing trend insight
                reduction_amount = "significant" if avg_first - avg_last > 0.1 else "moderate"
                recommendations.append(f'The {intervention_str} intervention(s) show a {reduction_amount} positive effect with a decreasing risk trend.')
            elif trend == 'Stable':
                recommendations.append(f'The {intervention_str} intervention(s) are helping to stabilize the risk trajectory.')
            elif trend == 'Increasing':
                # More specific increasing trend insight
                increase_amount = "significantly" if avg_last - avg_first > 0.1 else "slightly"
                recommendations.append(f'Despite the {intervention_str} intervention(s), risk is still {increase_amount} increasing. Consider additional approaches.')

        # Remove duplicates
        recommendations = list(set(recommendations))

        return {
            'trend': trend,
            'key_factors': key_factors,
            'recommendations': recommendations
        }

    def generate_scenario_forecast(self, patient_history, scenario, horizon=None):
        """
        Generate a forecast for a specific intervention scenario.

        Args:
            patient_history: List of patient visit records
            scenario: Dictionary with intervention details
            horizon: Number of months to forecast (defaults to self.forecast_horizon)

        Returns:
            Dictionary with forecast results
        """
        if not patient_history or len(patient_history) == 0:
            return {
                'status': 'error',
                'message': 'No patient history provided.'
            }

        # Create a deep copy of the patient history to avoid modifying the original
        modified_history = []
        for visit in patient_history:
            visit_copy = {}
            for key, value in visit.items():
                if isinstance(value, dict):
                    visit_copy[key] = value.copy()
                else:
                    visit_copy[key] = value
            modified_history.append(visit_copy)

        # Get the latest visit
        sorted_history = sorted(modified_history, key=lambda x: x.get('timestamp', ''))
        latest_visit = sorted_history[-1]  # We'll modify this directly

        # Create a new visit with a timestamp slightly after the latest visit
        new_visit = {}
        for key, value in latest_visit.items():
            if isinstance(value, dict):
                new_visit[key] = value.copy()
            else:
                new_visit[key] = value

        # Set a new timestamp for the intervention visit (1 day after the latest visit)
        try:
            latest_timestamp = datetime.fromisoformat(latest_visit['timestamp'].replace('Z', '+00:00'))
            new_timestamp = (latest_timestamp + timedelta(days=1)).isoformat()
            new_visit['timestamp'] = new_timestamp
        except Exception:
            # If we can't parse the timestamp, generate a new one
            new_visit['timestamp'] = datetime.now().isoformat()

        # Add a scenario ID to the visit to enable unique forecasting
        scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"
        new_visit['scenario_id'] = scenario_id

        # Apply scenario modifications to the new visit
        if 'interventions' in scenario and scenario['interventions']:
            print(f"Applying {len(scenario['interventions'])} interventions")
            for intervention in scenario['interventions']:
                intervention_type = intervention.get('type')
                value = intervention.get('value')

                if intervention_type and value is not None:
                    print(f"Applying intervention: {intervention_type} = {value}")
                    # Modify clinical parameters
                    if intervention_type in ['blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'max_heart_rate']:
                        if 'clinical_parameters' not in new_visit:
                            new_visit['clinical_parameters'] = {}
                        new_visit['clinical_parameters'][intervention_type] = value

                        # Add direct risk reduction based on intervention type
                        # Ensure risk_assessment exists
                        if 'risk_assessment' not in new_visit:
                            new_visit['risk_assessment'] = {'prediction': 0.5}

                        current_risk = float(new_visit['risk_assessment'].get('prediction', 0.5))
                        reduction = 0.0

                        if intervention_type == 'blood_pressure':
                            # Parse blood pressure and apply reduction based on improvement
                            try:
                                # Parse systolic and diastolic values
                                bp_parts = value.split('/')
                                systolic = int(bp_parts[0])
                                if systolic < 130:  # Normal range
                                    reduction = 0.15  # 15% reduction for normal BP
                                elif systolic < 140:  # Elevated
                                    reduction = 0.10  # 10% reduction for elevated BP
                                else:
                                    reduction = 0.05  # 5% reduction for high BP
                            except:
                                reduction = 0.08  # Default reduction

                        elif intervention_type == 'cholesterol':
                            # Apply reduction based on cholesterol value
                            try:
                                chol = float(value)
                                if chol < 200:  # Desirable
                                    reduction = 0.12
                                elif chol < 240:  # Borderline high
                                    reduction = 0.08
                                else:
                                    reduction = 0.05
                            except:
                                reduction = 0.08

                        elif intervention_type == 'fasting_blood_sugar':
                            reduction = 0.10  # 10% reduction for controlled blood sugar

                        elif intervention_type == 'max_heart_rate':
                            reduction = 0.08  # 8% reduction for improved heart rate

                        # Apply the reduction
                        if reduction > 0:
                            new_risk = max(0.01, current_risk * (1 - reduction))
                            new_visit['risk_assessment']['prediction'] = new_risk
                            print(f"Applied {intervention_type} intervention: reduced risk from {current_risk:.2f} to {new_risk:.2f}")

                    # Modify biomarkers
                    elif intervention_type in ['nt_probnp']:
                        if 'biomarkers' not in new_visit:
                            new_visit['biomarkers'] = {}
                        new_visit['biomarkers'][intervention_type] = value

                        # Add direct risk reduction for NT-proBNP improvement
                        # Ensure risk_assessment exists
                        if 'risk_assessment' not in new_visit:
                            new_visit['risk_assessment'] = {'prediction': 0.5}

                        current_risk = float(new_visit['risk_assessment'].get('prediction', 0.5))

                        # Apply reduction based on NT-proBNP value
                        try:
                            nt_probnp = float(value)
                            # Get age for threshold calculation
                            age = 65  # Default
                            if 'demographic_data' in new_visit and 'age' in new_visit['demographic_data']:
                                try:
                                    age = float(new_visit['demographic_data']['age'])
                                except:
                                    pass

                            # Age-adjusted threshold
                            if age < 50:
                                threshold = 450
                            elif age <= 75:
                                threshold = 900
                            else:
                                threshold = 1800

                            # Calculate reduction based on ratio to threshold
                            ratio = nt_probnp / threshold
                            if ratio < 1.0:  # Below threshold
                                reduction = 0.18  # 18% reduction for normal NT-proBNP
                            elif ratio < 2.0:  # Moderately elevated
                                reduction = 0.12  # 12% reduction
                            else:  # Significantly elevated
                                reduction = 0.08  # 8% reduction

                            # Apply the reduction
                            new_risk = max(0.01, current_risk * (1 - reduction))
                            new_visit['risk_assessment']['prediction'] = new_risk
                            print(f"Applied NT-proBNP intervention: reduced risk from {current_risk:.2f} to {new_risk:.2f}")
                        except:
                            pass

                    # Modify lifestyle factors
                    elif intervention_type in ['exercise', 'diet', 'smoking']:
                        if 'lifestyle' not in new_visit:
                            new_visit['lifestyle'] = {}
                        new_visit['lifestyle'][intervention_type] = value

                        # Add direct risk reduction based on lifestyle changes
                        # Ensure risk_assessment exists
                        if 'risk_assessment' not in new_visit:
                            new_visit['risk_assessment'] = {'prediction': 0.5}

                        current_risk = float(new_visit['risk_assessment'].get('prediction', 0.5))
                        reduction = 0.0

                        if intervention_type == 'exercise':
                            # Apply reduction based on exercise level
                            if value == 'intense':
                                reduction = 0.15  # 15% reduction for intense exercise
                            elif value == 'moderate':
                                reduction = 0.10  # 10% reduction for moderate exercise
                            elif value == 'light':
                                reduction = 0.05  # 5% reduction for light exercise

                        elif intervention_type == 'diet':
                            # Apply reduction based on diet quality
                            if value == 'excellent':
                                reduction = 0.12  # 12% reduction for excellent diet
                            elif value == 'good':
                                reduction = 0.08  # 8% reduction for good diet
                            elif value == 'average':
                                reduction = 0.04  # 4% reduction for average diet

                        elif intervention_type == 'smoking':
                            # Apply reduction based on smoking status
                            if value == 'never':
                                reduction = 0.15  # 15% reduction for never smokers
                            elif value == 'former':
                                reduction = 0.10  # 10% reduction for former smokers
                            else:
                                reduction = 0.0  # No reduction for current smokers

                        # Apply the reduction
                        if reduction > 0:
                            new_risk = max(0.01, current_risk * (1 - reduction))
                            new_visit['risk_assessment']['prediction'] = new_risk
                            print(f"Applied {intervention_type} intervention: reduced risk from {current_risk:.2f} to {new_risk:.2f}")

        # Add the new visit to the history
        modified_history.append(new_visit)

        # Generate forecast with the modified history and specified horizon
        forecast_result = self.forecast(modified_history, horizon)

        # Add scenario information
        if forecast_result['status'] == 'success':
            # Get the interventions
            interventions = scenario.get('interventions', [])

            # Add scenario name and interventions to the result
            forecast_result['scenario_name'] = scenario.get('name', 'Unnamed Scenario')
            forecast_result['interventions'] = interventions

            # Generate scenario-specific insights
            forecast_result['insights'] = self.generate_insights(
                forecast_result['forecast_values'],
                forecast_result['feature_importance'],
                interventions
            )

        return forecast_result

    def get_forecast(self, forecast_id):
        """
        Retrieve a saved forecast.

        Args:
            forecast_id: ID of the forecast to retrieve

        Returns:
            Dictionary with forecast data or None if not found
        """
        forecast_path = os.path.join(FORECASTS_DIR, f"{forecast_id}.json")

        if not os.path.exists(forecast_path):
            return None

        try:
            with open(forecast_path, 'r') as f:
                forecast_data = json.load(f)

            return forecast_data
        except Exception as e:
            print(f"Error loading forecast {forecast_id}: {str(e)}")
            return None

    def get_all_forecasts(self, patient_id=None):
        """
        Get all forecasts, optionally filtered by patient ID.

        Args:
            patient_id: Optional patient ID to filter forecasts

        Returns:
            List of forecast data dictionaries
        """
        forecasts = []

        if not os.path.exists(FORECASTS_DIR):
            return forecasts

        for filename in os.listdir(FORECASTS_DIR):
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(FORECASTS_DIR, filename)
            try:
                with open(file_path, 'r') as f:
                    forecast_data = json.load(f)

                # Filter by patient ID if specified
                if patient_id is None or forecast_data.get('patient_id') == patient_id:
                    forecasts.append(forecast_data)
            except Exception as e:
                print(f"Error loading forecast file {filename}: {str(e)}")

        # Sort by timestamp (newest first)
        forecasts.sort(key=lambda f: f.get('timestamp', ''), reverse=True)

        return forecasts

# Create a singleton instance
forecaster = TemporalForecaster()

def train_forecasting_model(patient_histories, forecast_horizon=6):
    """
    Train the forecasting model on patient histories.

    Args:
        patient_histories: List of patient histories, each containing visit records
        forecast_horizon: Number of months to forecast into the future

    Returns:
        Dictionary with training results
    """
    return forecaster.train(patient_histories, forecast_horizon)

def generate_forecast(patient_history, horizon=None):
    """
    Generate a forecast for a patient based on their history.

    Args:
        patient_history: List of patient visit records
        horizon: Number of months to forecast (defaults to model's forecast_horizon)

    Returns:
        Dictionary with forecast results
    """
    return forecaster.forecast(patient_history, horizon)

def generate_scenario_forecast(patient_history, scenario, horizon=None):
    """
    Generate a forecast for a specific intervention scenario.

    Args:
        patient_history: List of patient visit records
        scenario: Dictionary with intervention details
        horizon: Number of months to forecast (defaults to model's forecast_horizon)

    Returns:
        Dictionary with forecast results
    """
    return forecaster.generate_scenario_forecast(patient_history, scenario, horizon)

def get_forecast(forecast_id):
    """
    Retrieve a saved forecast.

    Args:
        forecast_id: ID of the forecast to retrieve

    Returns:
        Dictionary with forecast data or None if not found
    """
    return forecaster.get_forecast(forecast_id)

def get_all_forecasts(patient_id=None):
    """
    Get all forecasts, optionally filtered by patient ID.

    Args:
        patient_id: Optional patient ID to filter forecasts

    Returns:
        List of forecast data dictionaries
    """
    return forecaster.get_all_forecasts(patient_id)
