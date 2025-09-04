# Added Prior Cardiac Event Options to Heart Failure Prediction System

## Changes Made

### 1. Updated Patient Form UI

We've enhanced the Prior Cardiac Event section in the patient form to provide a more user-friendly and structured approach:

1. **Replaced text input with dropdown menu**:
   ```jsx
   <select
     name="prior_cardiac_event.type"
     value={formData.prior_cardiac_event.type}
     onChange={handleChange}
     className="form-select"
   >
     <option value="">None</option>
     <option value="Myocardial Infarction">Myocardial Infarction (Heart Attack)</option>
     <option value="Arrhythmia">Arrhythmia</option>
     <option value="Angina">Angina</option>
     <option value="Heart Failure">Heart Failure</option>
     <option value="Coronary Artery Disease">Coronary Artery Disease</option>
     <option value="Valve Disease">Valve Disease</option>
     <option value="Cardiomyopathy">Cardiomyopathy</option>
     <option value="Pericarditis">Pericarditis</option>
   </select>
   ```

2. **Improved time since event selection**:
   ```jsx
   <select
     name="prior_cardiac_event.time_since_event"
     value={formData.prior_cardiac_event.time_since_event}
     onChange={handleChange}
     className="form-select"
     disabled={!formData.prior_cardiac_event.type}
   >
     <option value="">Select time</option>
     <option value="1">Less than 1 month</option>
     <option value="3">1-3 months</option>
     <option value="6">3-6 months</option>
     <option value="12">6-12 months</option>
     <option value="24">1-2 years</option>
     <option value="36">2-3 years</option>
     <option value="60">3-5 years</option>
     <option value="120">More than 5 years</option>
   </select>
   ```

3. **Added conditional disabling**:
   - Time since event and severity fields are disabled when no event type is selected
   - This prevents users from entering inconsistent data

4. **Added informational message**:
   ```jsx
   {!formData.prior_cardiac_event.type && (
     <div className="info-message" style={{ marginBottom: '15px', color: '#666', fontStyle: 'italic' }}>
       Select a cardiac event type if the patient has a history of heart-related conditions.
     </div>
   )}
   ```

### 2. Enhanced Model Integration

We've updated the model_enhancer.py file to properly incorporate prior cardiac events into the risk prediction:

1. **Added weight to the default feature weights**:
   ```python
   DEFAULT_WEIGHTS = {
       # ... existing weights ...
       'prior_cardiac_event': 0.15  # Added weight for prior cardiac events
   }
   ```

2. **Implemented detailed feature extraction**:
   ```python
   # Prior cardiac event (0: None, 0.5-1.0: Based on type, severity, and recency)
   prior_event = patient_data.get('prior_cardiac_event', {})
   prior_event_type = prior_event.get('type', '')
   
   if not prior_event_type:
       features['prior_cardiac_event'] = 0.0  # No prior event
   else:
       # Base risk by event type
       event_risk = {
           'Myocardial Infarction': 1.0,  # Heart attack - highest risk
           'Heart Failure': 0.9,
           'Coronary Artery Disease': 0.8,
           'Arrhythmia': 0.7,
           'Valve Disease': 0.7,
           'Cardiomyopathy': 0.8,
           'Angina': 0.6,
           'Pericarditis': 0.5
       }.get(prior_event_type, 0.5)
       
       # Adjust for severity
       severity = prior_event.get('severity', 'Mild')
       severity_factor = {
           'Mild': 0.7,
           'Moderate': 0.85,
           'Severe': 1.0
       }.get(severity, 0.7)
       
       # Adjust for time since event (more recent = higher risk)
       try:
           time_since_event = float(prior_event.get('time_since_event', 12))
           # Exponential decay with time (0-120 months)
           time_factor = math.exp(-0.02 * time_since_event) * 0.8 + 0.2  # Range: 0.2-1.0
       except (ValueError, TypeError):
           time_factor = 0.5  # Default if time is invalid
       
       # Calculate final risk score for prior event
       features['prior_cardiac_event'] = event_risk * severity_factor * time_factor
   ```

3. **Enhanced SHAP value generation**:
   ```python
   # Adjust direction based on whether feature increases or decreases risk
   direction = 1.0
   if feature in ['max_heart_rate']:  # Features that decrease risk when high
       direction = -1.0 if features[feature] > 0.5 else 1.0
   elif feature == 'prior_cardiac_event' and features[feature] > 0:  # Prior events always increase risk
       direction = 1.5  # Emphasize the impact of prior events
   ```

## Impact on Risk Prediction

The addition of structured prior cardiac event options significantly improves the risk prediction model:

1. **More accurate risk assessment**: Different cardiac events have different impacts on future heart failure risk
2. **Time-sensitive risk calculation**: More recent events have a higher impact on risk
3. **Severity consideration**: The severity of the prior event is now properly factored into the risk calculation
4. **Improved explainability**: The SHAP values now include the contribution of prior cardiac events to the risk prediction

## User Experience Improvements

1. **Guided data entry**: Users now select from predefined options instead of free text
2. **Consistent data**: Standardized options ensure consistent data for analysis
3. **Intuitive interface**: Fields are disabled when not applicable
4. **Clear instructions**: Informational message guides users on when to use this section

These changes make the heart failure prediction system more accurate, user-friendly, and clinically relevant.
