"""
Rule-based model for heart failure prediction
"""

import numpy as np
import config

class RuleBasedModel:
    """
    Implements a rule-based clinical model for heart failure risk assessment
    based on established clinical guidelines
    """
    
    def __init__(self):
        """Initialize the rule-based model with clinical thresholds"""
        self.thresholds = config.MODEL_CONFIG['rule_based']
        self.nt_probnp_thresholds = config.NT_PROBNP_THRESHOLDS
        
    def predict(self, features):
        """
        Predict heart failure risk using rule-based clinical guidelines
        
        Args:
            features: DataFrame containing patient features
            
        Returns:
            Array of risk scores between 0 and 1
        """
        # Initialize risk scores
        risk_scores = np.zeros(len(features))
        confidence_scores = np.ones(len(features)) * 0.7  # Base confidence
        
        # Apply rules for each patient
        for i, (_, patient) in enumerate(features.iterrows()):
            risk_score, confidence = self._apply_rules(patient)
            risk_scores[i] = risk_score
            confidence_scores[i] = confidence
            
        return risk_scores, confidence_scores
    
    def _apply_rules(self, patient):
        """
        Apply clinical rules to a single patient
        
        Args:
            patient: Series containing patient features
            
        Returns:
            risk_score: Risk score between 0 and 1
            confidence: Confidence in the prediction
        """
        risk_score = 0.0
        rule_count = 0
        confidence = 0.7  # Base confidence
        
        # Age rule
        if 'age' in patient:
            if patient['age'] > self.thresholds['age_threshold']:
                risk_score += 0.1
                rule_count += 1
                
        # Blood pressure rule
        if 'systolic_bp' in patient and 'diastolic_bp' in patient:
            if patient['systolic_bp'] > self.thresholds['bp_systolic_threshold']:
                risk_score += 0.15
                rule_count += 1
            if patient['diastolic_bp'] > self.thresholds['bp_diastolic_threshold']:
                risk_score += 0.1
                rule_count += 1
                
        # Cholesterol rule
        if 'cholesterol' in patient:
            if patient['cholesterol'] > self.thresholds['cholesterol_threshold']:
                risk_score += 0.1
                rule_count += 1
                
        # Fasting blood sugar rule
        if 'fasting_blood_sugar' in patient:
            if patient['fasting_blood_sugar'] > self.thresholds['fasting_glucose_threshold']:
                risk_score += 0.1
                rule_count += 1
                
        # Max heart rate rule (lower is worse)
        if 'max_heart_rate' in patient:
            if patient['max_heart_rate'] < self.thresholds['max_heart_rate_threshold']:
                risk_score += 0.1
                rule_count += 1
                
        # ST depression rule
        if 'st_depression' in patient:
            if patient['st_depression'] > self.thresholds['st_depression_threshold']:
                risk_score += 0.15
                rule_count += 1
                
        # Number of vessels rule
        if 'num_vessels' in patient:
            if patient['num_vessels'] > self.thresholds['num_vessels_threshold']:
                risk_score += 0.15 * patient['num_vessels']
                rule_count += 1
                
        # NT-proBNP rule with age adjustment
        if 'nt_probnp' in patient and 'age' in patient:
            threshold = self._get_nt_probnp_threshold(patient['age'])
            if patient['nt_probnp'] > threshold:
                # Higher excess over threshold means higher risk
                excess_ratio = patient['nt_probnp'] / threshold
                risk_score += min(0.25, 0.15 * excess_ratio)
                rule_count += 1
                confidence += 0.1  # Higher confidence with biomarker data
                
        # Prior cardiac event rule
        if 'prior_cardiac_event' in patient:
            if patient['prior_cardiac_event'] > 0:
                risk_score += 0.2
                rule_count += 1
                
                # If we have time since event, adjust based on recency
                if 'time_since_event' in patient:
                    # More recent events have higher impact
                    time_factor = np.exp(-0.001 * patient['time_since_event'])
                    risk_score += 0.1 * time_factor
                    
        # ECG abnormalities rules
        if 'ecg_af_detected' in patient and patient['ecg_af_detected'] > 0:
            risk_score += 0.15
            rule_count += 1
            confidence += 0.05
            
        if 'ecg_pvc_count' in patient and patient['ecg_pvc_count'] > 0:
            risk_score += 0.05 * min(3, patient['ecg_pvc_count'])
            rule_count += 1
            confidence += 0.05
            
        if 'ecg_qt_prolongation' in patient and patient['ecg_qt_prolongation'] > 0:
            risk_score += 0.1
            rule_count += 1
            confidence += 0.05
            
        # Adjust confidence based on number of rules applied
        if rule_count > 5:
            confidence += 0.1
        elif rule_count < 3:
            confidence -= 0.1
            
        # Normalize risk score to 0-1 range
        risk_score = min(1.0, risk_score)
        
        # Ensure confidence is in 0-1 range
        confidence = max(0.5, min(0.9, confidence))
        
        return risk_score, confidence
    
    def _get_nt_probnp_threshold(self, age):
        """Get age-adjusted NT-proBNP threshold"""
        if age < 50:
            return self.nt_probnp_thresholds['age_lt_50']
        elif age <= 75:
            return self.nt_probnp_thresholds['age_50_to_75']
        else:
            return self.nt_probnp_thresholds['age_gt_75']
    
    def explain(self, patient):
        """
        Generate explanation for rule-based prediction
        
        Args:
            patient: Series containing patient features
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'factors': [],
            'summary': '',
            'confidence': 0
        }
        
        # Apply rules and track contributions
        risk_score, confidence = self._apply_rules(patient)
        explanation['confidence'] = confidence
        
        # Age factor
        if 'age' in patient and patient['age'] > self.thresholds['age_threshold']:
            explanation['factors'].append({
                'name': 'Age',
                'value': patient['age'],
                'contribution': 0.1,
                'description': f"Age > {self.thresholds['age_threshold']} is a risk factor for heart failure."
            })
            
        # Blood pressure factor
        if 'systolic_bp' in patient and patient['systolic_bp'] > self.thresholds['bp_systolic_threshold']:
            explanation['factors'].append({
                'name': 'Systolic Blood Pressure',
                'value': patient['systolic_bp'],
                'contribution': 0.15,
                'description': f"Systolic BP > {self.thresholds['bp_systolic_threshold']} indicates hypertension."
            })
            
        if 'diastolic_bp' in patient and patient['diastolic_bp'] > self.thresholds['bp_diastolic_threshold']:
            explanation['factors'].append({
                'name': 'Diastolic Blood Pressure',
                'value': patient['diastolic_bp'],
                'contribution': 0.1,
                'description': f"Diastolic BP > {self.thresholds['bp_diastolic_threshold']} indicates hypertension."
            })
            
        # Cholesterol factor
        if 'cholesterol' in patient and patient['cholesterol'] > self.thresholds['cholesterol_threshold']:
            explanation['factors'].append({
                'name': 'Cholesterol',
                'value': patient['cholesterol'],
                'contribution': 0.1,
                'description': f"Cholesterol > {self.thresholds['cholesterol_threshold']} increases cardiovascular risk."
            })
            
        # NT-proBNP factor
        if 'nt_probnp' in patient and 'age' in patient:
            threshold = self._get_nt_probnp_threshold(patient['age'])
            if patient['nt_probnp'] > threshold:
                excess_ratio = patient['nt_probnp'] / threshold
                contribution = min(0.25, 0.15 * excess_ratio)
                explanation['factors'].append({
                    'name': 'NT-proBNP',
                    'value': patient['nt_probnp'],
                    'contribution': contribution,
                    'description': f"NT-proBNP > {threshold} pg/mL (age-adjusted threshold) indicates cardiac stress."
                })
                
        # Prior cardiac event factor
        if 'prior_cardiac_event' in patient and patient['prior_cardiac_event'] > 0:
            contribution = 0.2
            if 'time_since_event' in patient:
                time_factor = np.exp(-0.001 * patient['time_since_event'])
                contribution += 0.1 * time_factor
                
            explanation['factors'].append({
                'name': 'Prior Cardiac Event',
                'value': 'Yes',
                'contribution': contribution,
                'description': "Previous cardiac events significantly increase heart failure risk."
            })
            
        # ECG abnormalities
        if 'ecg_af_detected' in patient and patient['ecg_af_detected'] > 0:
            explanation['factors'].append({
                'name': 'Atrial Fibrillation',
                'value': 'Detected',
                'contribution': 0.15,
                'description': "Atrial fibrillation increases heart failure risk."
            })
            
        # Sort factors by contribution
        explanation['factors'].sort(key=lambda x: x['contribution'], reverse=True)
        
        # Generate summary
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.3:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
            
        top_factors = [f['name'] for f in explanation['factors'][:3]]
        if top_factors:
            factors_text = ", ".join(top_factors)
            explanation['summary'] = f"Rule-based assessment indicates {risk_level} heart failure risk primarily due to {factors_text}."
        else:
            explanation['summary'] = f"Rule-based assessment indicates {risk_level} heart failure risk."
            
        return explanation
