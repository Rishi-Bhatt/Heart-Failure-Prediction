"""
Counterfactual Explanation Engine for Heart Failure Prediction

This module provides counterfactual explanations for heart failure predictions,
showing how changes to specific risk factors would affect the predicted risk.

References:
1. Wachter, S., Mittelstadt, B., & Russell, C. (2017). "Counterfactual Explanations
   Without Opening the Black Box: Automated Decisions and the GDPR"
2. Mothilal, R. K., Sharma, A., & Tan, C. (2020). "Explaining machine learning
   classifiers through diverse counterfactual explanations"
"""

import os
import json
import numpy as np
import copy
from datetime import datetime
import clinical_ml_model
import model_enhancer
import hybrid_model

# Constants for counterfactual generation
NUMERICAL_FEATURES = [
    'age', 'cholesterol', 'blood_pressure_systolic', 'blood_pressure_diastolic',
    'fasting_blood_sugar', 'max_heart_rate', 'st_depression'
]

CATEGORICAL_FEATURES = [
    'gender', 'chest_pain_type', 'exercise_induced_angina', 'st_slope',
    'num_major_vessels', 'thalassemia'
]

BIOMARKER_FEATURES = [
    'nt_probnp'
]

# Clinical guidelines for feature modifications
CLINICAL_GUIDELINES = {
    'cholesterol': {
        'target_range': (0, 200),
        'improvement_text': 'Reduce cholesterol to below 200 mg/dL through diet, exercise, or medication',
        'clinical_reference': 'ACC/AHA Guidelines on the Treatment of Blood Cholesterol (2018)',
        'intervention_difficulty': 'moderate',
        'typical_improvement': 30  # Typical improvement with intervention (mg/dL)
    },
    'blood_pressure_systolic': {
        'target_range': (90, 120),
        'improvement_text': 'Reduce systolic blood pressure to below 120 mmHg through lifestyle changes or medication',
        'clinical_reference': 'ACC/AHA Guidelines for Hypertension (2017)',
        'intervention_difficulty': 'moderate',
        'typical_improvement': 10  # Typical improvement with intervention (mmHg)
    },
    'blood_pressure_diastolic': {
        'target_range': (60, 80),
        'improvement_text': 'Reduce diastolic blood pressure to below 80 mmHg through lifestyle changes or medication',
        'clinical_reference': 'ACC/AHA Guidelines for Hypertension (2017)',
        'intervention_difficulty': 'moderate',
        'typical_improvement': 5  # Typical improvement with intervention (mmHg)
    },
    'fasting_blood_sugar': {
        'target_range': (70, 100),
        'improvement_text': 'Maintain fasting blood sugar below 100 mg/dL through diet, exercise, and medication if needed',
        'clinical_reference': 'ADA Standards of Medical Care in Diabetes (2021)',
        'intervention_difficulty': 'moderate',
        'typical_improvement': 20  # Typical improvement with intervention (mg/dL)
    },
    'max_heart_rate': {
        'target_range': (60, 100),
        'improvement_text': 'Maintain resting heart rate between 60-100 bpm through regular exercise and stress management',
        'clinical_reference': 'ESC Guidelines on Cardiovascular Disease Prevention (2016)',
        'intervention_difficulty': 'easy',
        'typical_improvement': 10  # Typical improvement with intervention (bpm)
    },
    'exercise_induced_angina': {
        'target_value': 0,
        'improvement_text': 'Reduce exercise-induced angina through cardiac rehabilitation and medication',
        'clinical_reference': 'ACC/AHA Guidelines for Stable Ischemic Heart Disease (2014)',
        'intervention_difficulty': 'hard',
        'typical_improvement': 1  # Binary improvement (presence to absence)
    },
    'st_depression': {
        'target_range': (0, 1.0),
        'improvement_text': 'Reduce ST depression through medication and cardiac rehabilitation',
        'clinical_reference': 'ACC/AHA Guidelines for Stable Ischemic Heart Disease (2014)',
        'intervention_difficulty': 'hard',
        'typical_improvement': 0.5  # Typical improvement with intervention (mm)
    },
    'nt_probnp': {
        'target_range': (0, 125),  # For patients < 75 years
        'target_range_elderly': (0, 450),  # For patients >= 75 years
        'improvement_text': 'Reduce NT-proBNP levels through heart failure medication and lifestyle changes',
        'clinical_reference': 'ESC Guidelines for Heart Failure (2016)',
        'intervention_difficulty': 'hard',
        'typical_improvement': 100  # Typical improvement with intervention (pg/mL)
    }
}

class CounterfactualEngine:
    """
    Engine for generating counterfactual explanations for heart failure predictions.
    """

    def __init__(self):
        """Initialize the counterfactual engine."""
        # Use the hybrid model for predictions
        self.predictor = hybrid_model.hybrid_model

    def generate_counterfactuals(self, patient_data, num_counterfactuals=5):
        """
        Generate counterfactual explanations for a patient.

        Args:
            patient_data: Dictionary containing patient data
            num_counterfactuals: Number of counterfactual scenarios to generate

        Returns:
            Dictionary with counterfactual explanations
        """
        # Get the original prediction
        original_prediction, original_confidence, _ = self.predictor.predict(patient_data)

        # Generate individual feature counterfactuals
        feature_counterfactuals = self._generate_feature_counterfactuals(patient_data)

        # Generate combined counterfactuals (realistic intervention scenarios)
        combined_counterfactuals = self._generate_combined_counterfactuals(
            patient_data, feature_counterfactuals, num_counterfactuals
        )

        # Add statistical confidence intervals
        for cf in combined_counterfactuals:
            cf['confidence_interval'] = self._calculate_confidence_interval(
                cf['modified_prediction'], original_confidence
            )

        # Sort counterfactuals by impact (largest risk reduction first)
        feature_counterfactuals.sort(
            key=lambda x: original_prediction - x['modified_prediction'],
            reverse=True
        )

        combined_counterfactuals.sort(
            key=lambda x: original_prediction - x['modified_prediction'],
            reverse=True
        )

        return {
            'original_prediction': float(original_prediction),
            'original_confidence': float(original_confidence),
            'feature_counterfactuals': feature_counterfactuals[:num_counterfactuals],
            'combined_counterfactuals': combined_counterfactuals[:num_counterfactuals],
            'clinical_guidelines': CLINICAL_GUIDELINES,
            'timestamp': datetime.now().isoformat()
        }

    def _generate_feature_counterfactuals(self, patient_data):
        """
        Generate counterfactuals by modifying individual features.

        Args:
            patient_data: Dictionary containing patient data

        Returns:
            List of counterfactual scenarios
        """
        original_prediction, _, _ = self.predictor.predict(patient_data)
        counterfactuals = []

        # Process numerical features
        for feature in NUMERICAL_FEATURES:
            if feature in patient_data:
                # Skip if feature is already in optimal range
                if feature in CLINICAL_GUIDELINES:
                    guideline = CLINICAL_GUIDELINES[feature]
                    target_range = guideline['target_range']

                    # Skip if already in target range
                    if target_range[0] <= float(patient_data[feature]) <= target_range[1]:
                        continue

                # Create modified patient data
                modified_data = copy.deepcopy(patient_data)

                # Get original value
                try:
                    original_value = float(patient_data[feature]) if patient_data[feature] != '' else 0
                except (ValueError, TypeError):
                    # Skip features with invalid values
                    continue

                # Determine improved value based on clinical guidelines
                if feature in CLINICAL_GUIDELINES:
                    guideline = CLINICAL_GUIDELINES[feature]

                    # Use typical improvement from guidelines
                    if 'typical_improvement' in guideline:
                        if original_value > guideline['target_range'][1]:
                            # Need to decrease
                            modified_value = max(
                                original_value - guideline['typical_improvement'],
                                guideline['target_range'][1]
                            )
                        elif original_value < guideline['target_range'][0]:
                            # Need to increase
                            modified_value = min(
                                original_value + guideline['typical_improvement'],
                                guideline['target_range'][0]
                            )
                        else:
                            # Already in range, use midpoint
                            modified_value = sum(guideline['target_range']) / 2
                    else:
                        # Use midpoint of target range
                        modified_value = sum(guideline['target_range']) / 2
                else:
                    # Default improvement (20% better)
                    modified_value = original_value * 0.8

                # Update the feature
                modified_data[feature] = modified_value

                # Get new prediction
                modified_prediction, modified_confidence, _ = self.predictor.predict(modified_data)

                # Calculate impact
                absolute_impact = original_prediction - modified_prediction
                relative_impact = (absolute_impact / original_prediction) * 100

                # Create counterfactual explanation
                counterfactual = {
                    'feature': feature,
                    'original_value': original_value,
                    'modified_value': modified_value,
                    'original_prediction': float(original_prediction),
                    'modified_prediction': float(modified_prediction),
                    'absolute_impact': float(absolute_impact),
                    'relative_impact': float(relative_impact),
                    'confidence': float(modified_confidence)
                }

                # Add clinical guideline information if available
                if feature in CLINICAL_GUIDELINES:
                    counterfactual['clinical_guideline'] = CLINICAL_GUIDELINES[feature]

                counterfactuals.append(counterfactual)

        # Process categorical features
        for feature in CATEGORICAL_FEATURES:
            if feature in patient_data and feature in CLINICAL_GUIDELINES:
                # Only process categorical features with clinical guidelines
                guideline = CLINICAL_GUIDELINES[feature]

                # Skip if already at target value
                if patient_data[feature] == guideline.get('target_value'):
                    continue

                # Create modified patient data
                modified_data = copy.deepcopy(patient_data)

                # Get original value
                original_value = patient_data[feature]

                # Set to target value
                modified_data[feature] = guideline['target_value']

                # Get new prediction
                modified_prediction, modified_confidence, _ = self.predictor.predict(modified_data)

                # Calculate impact
                absolute_impact = original_prediction - modified_prediction
                relative_impact = (absolute_impact / original_prediction) * 100

                # Create counterfactual explanation
                counterfactual = {
                    'feature': feature,
                    'original_value': original_value,
                    'modified_value': guideline['target_value'],
                    'original_prediction': float(original_prediction),
                    'modified_prediction': float(modified_prediction),
                    'absolute_impact': float(absolute_impact),
                    'relative_impact': float(relative_impact),
                    'confidence': float(modified_confidence),
                    'clinical_guideline': guideline
                }

                counterfactuals.append(counterfactual)

        # Process biomarker features
        for feature in BIOMARKER_FEATURES:
            if feature in patient_data.get('biomarkers', {}):
                # Create modified patient data
                modified_data = copy.deepcopy(patient_data)

                # Get original value
                try:
                    biomarker_value = patient_data['biomarkers'][feature]
                    original_value = float(biomarker_value) if biomarker_value != '' else 0
                except (ValueError, TypeError, KeyError):
                    # Skip features with invalid values
                    continue

                # Determine improved value based on clinical guidelines
                if feature in CLINICAL_GUIDELINES:
                    guideline = CLINICAL_GUIDELINES[feature]

                    # Use age-specific target range for NT-proBNP
                    if feature == 'nt_probnp':
                        if int(patient_data.get('age', 0)) >= 75:
                            target_range = guideline.get('target_range_elderly', guideline['target_range'])
                        else:
                            target_range = guideline['target_range']
                    else:
                        target_range = guideline['target_range']

                    # Skip if already in target range
                    if target_range[0] <= original_value <= target_range[1]:
                        continue

                    # Use typical improvement from guidelines
                    if 'typical_improvement' in guideline:
                        if original_value > target_range[1]:
                            # Need to decrease
                            modified_value = max(
                                original_value - guideline['typical_improvement'],
                                target_range[1]
                            )
                        elif original_value < target_range[0]:
                            # Need to increase
                            modified_value = min(
                                original_value + guideline['typical_improvement'],
                                target_range[0]
                            )
                        else:
                            # Already in range, use midpoint
                            modified_value = sum(target_range) / 2
                    else:
                        # Use midpoint of target range
                        modified_value = sum(target_range) / 2
                else:
                    # Default improvement (20% better)
                    modified_value = original_value * 0.8

                # Update the feature
                modified_data['biomarkers'] = copy.deepcopy(patient_data.get('biomarkers', {}))
                modified_data['biomarkers'][feature] = modified_value

                # Get new prediction
                modified_prediction, modified_confidence, _ = self.predictor.predict(modified_data)

                # Calculate impact
                absolute_impact = original_prediction - modified_prediction
                relative_impact = (absolute_impact / original_prediction) * 100

                # Create counterfactual explanation
                counterfactual = {
                    'feature': f"biomarkers.{feature}",
                    'original_value': original_value,
                    'modified_value': modified_value,
                    'original_prediction': float(original_prediction),
                    'modified_prediction': float(modified_prediction),
                    'absolute_impact': float(absolute_impact),
                    'relative_impact': float(relative_impact),
                    'confidence': float(modified_confidence)
                }

                # Add clinical guideline information if available
                if feature in CLINICAL_GUIDELINES:
                    counterfactual['clinical_guideline'] = CLINICAL_GUIDELINES[feature]

                counterfactuals.append(counterfactual)

        return counterfactuals

    def _generate_combined_counterfactuals(self, patient_data, feature_counterfactuals, num_scenarios=3):
        """
        Generate combined counterfactual scenarios (multiple features changed together).

        Args:
            patient_data: Dictionary containing patient data
            feature_counterfactuals: List of individual feature counterfactuals
            num_scenarios: Number of combined scenarios to generate

        Returns:
            List of combined counterfactual scenarios
        """
        # Sort feature counterfactuals by impact
        sorted_counterfactuals = sorted(
            feature_counterfactuals,
            key=lambda x: x['absolute_impact'],
            reverse=True
        )

        # Generate combined scenarios
        combined_scenarios = []

        # Scenario 1: Top 2 most impactful features
        if len(sorted_counterfactuals) >= 2:
            scenario = self._create_combined_scenario(
                patient_data,
                [sorted_counterfactuals[0], sorted_counterfactuals[1]],
                "Highest Impact Intervention"
            )
            combined_scenarios.append(scenario)

        # Scenario 2: Easiest to implement features (based on intervention difficulty)
        easy_counterfactuals = [
            cf for cf in feature_counterfactuals
            if 'clinical_guideline' in cf and
            cf['clinical_guideline'].get('intervention_difficulty') == 'easy'
        ]

        if len(easy_counterfactuals) >= 2:
            # Sort by impact within easy interventions
            easy_counterfactuals.sort(key=lambda x: x['absolute_impact'], reverse=True)
            scenario = self._create_combined_scenario(
                patient_data,
                easy_counterfactuals[:2],
                "Easiest Intervention"
            )
            combined_scenarios.append(scenario)

        # Scenario 3: Clinically recommended combination
        # For heart failure, this often means addressing blood pressure, cholesterol, and exercise
        clinical_features = ['blood_pressure_systolic', 'cholesterol', 'max_heart_rate']
        clinical_counterfactuals = [
            cf for cf in feature_counterfactuals
            if cf['feature'] in clinical_features or
            (cf['feature'].startswith('biomarkers.') and cf['feature'].split('.')[1] in BIOMARKER_FEATURES)
        ]

        if len(clinical_counterfactuals) >= 2:
            scenario = self._create_combined_scenario(
                patient_data,
                clinical_counterfactuals[:3],
                "Clinically Recommended Intervention"
            )
            combined_scenarios.append(scenario)

        # Scenario 4: Comprehensive intervention (all modifiable factors)
        modifiable_counterfactuals = [
            cf for cf in feature_counterfactuals
            if cf['feature'] != 'age' and cf['feature'] != 'gender'
        ]

        if len(modifiable_counterfactuals) >= 3:
            scenario = self._create_combined_scenario(
                patient_data,
                modifiable_counterfactuals,
                "Comprehensive Intervention"
            )
            combined_scenarios.append(scenario)

        return combined_scenarios[:num_scenarios]

    def _create_combined_scenario(self, patient_data, counterfactuals, name):
        """
        Create a combined counterfactual scenario by applying multiple feature changes.

        Args:
            patient_data: Original patient data
            counterfactuals: List of counterfactuals to combine
            name: Name of the combined scenario

        Returns:
            Combined counterfactual scenario
        """
        # Create modified patient data
        modified_data = copy.deepcopy(patient_data)

        # Apply all modifications
        modified_features = []
        for cf in counterfactuals:
            feature = cf['feature']
            modified_value = cf['modified_value']

            # Handle biomarker features
            if feature.startswith('biomarkers.'):
                biomarker = feature.split('.')[1]
                if 'biomarkers' not in modified_data:
                    modified_data['biomarkers'] = {}
                modified_data['biomarkers'][biomarker] = modified_value
            else:
                # Regular feature
                modified_data[feature] = modified_value

            modified_features.append({
                'feature': feature,
                'original_value': cf['original_value'],
                'modified_value': modified_value,
                'absolute_impact': cf['absolute_impact'],
                'relative_impact': cf['relative_impact']
            })

        # Get original and new predictions
        original_prediction, original_confidence, _ = self.predictor.predict(patient_data)
        modified_prediction, modified_confidence, _ = self.predictor.predict(modified_data)

        # Calculate overall impact
        absolute_impact = original_prediction - modified_prediction
        relative_impact = (absolute_impact / original_prediction) * 100

        # Create combined scenario
        return {
            'name': name,
            'modified_features': modified_features,
            'original_prediction': float(original_prediction),
            'modified_prediction': float(modified_prediction),
            'absolute_impact': float(absolute_impact),
            'relative_impact': float(relative_impact),
            'confidence': float(modified_confidence)
        }

    def _calculate_confidence_interval(self, prediction, confidence, confidence_level=0.95):
        """
        Calculate confidence interval for a prediction.

        Args:
            prediction: Prediction value
            confidence: Confidence value
            confidence_level: Desired confidence level (default: 0.95 for 95% CI)

        Returns:
            Dictionary with lower and upper bounds
        """
        # Convert confidence to standard error
        # This is a simplified approach - in a real system, this would be more sophisticated
        std_error = (1 - confidence) / 2

        # Calculate z-score for the desired confidence level
        # For 95% CI, z = 1.96
        z_score = 1.96 if confidence_level == 0.95 else 1.645 if confidence_level == 0.90 else 2.576

        # Calculate margin of error
        margin_of_error = z_score * std_error

        # Calculate bounds
        lower_bound = max(0, prediction - margin_of_error)
        upper_bound = min(1, prediction + margin_of_error)

        return {
            'lower_bound': float(lower_bound),
            'upper_bound': float(upper_bound),
            'confidence_level': confidence_level
        }

# Create a singleton instance
counterfactual_engine = CounterfactualEngine()

def generate_counterfactuals(patient_data, num_counterfactuals=5):
    """
    Generate counterfactual explanations for a patient.

    Args:
        patient_data: Dictionary containing patient data
        num_counterfactuals: Number of counterfactual scenarios to generate

    Returns:
        Dictionary with counterfactual explanations
    """
    return counterfactual_engine.generate_counterfactuals(patient_data, num_counterfactuals)
