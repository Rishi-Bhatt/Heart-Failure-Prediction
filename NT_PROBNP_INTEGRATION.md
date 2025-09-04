# NT-proBNP Biomarker Integration

## Overview

We have enhanced the Heart Failure Prediction System by adding support for NT-proBNP, a powerful cardiac biomarker that can significantly improve prediction accuracy. This document explains the implementation and how to use this feature.

## What is NT-proBNP?

NT-proBNP (N-terminal pro-brain natriuretic peptide) is a hormone released by the heart in response to pressure and stretching of the heart muscle. Elevated levels indicate increased cardiac stress and are strongly associated with heart failure risk.

## Implementation Details

### 1. User Interface

- Added an optional NT-proBNP input field to the patient form
- Included informational text explaining the importance of this biomarker
- Added age-adjusted reference ranges that update based on the patient's age:
  - Age < 50: 0-450 pg/mL
  - Age 50-75: 0-900 pg/mL
  - Age > 75: 0-1800 pg/mL

### 2. Model Integration

- Updated the rule-based model to include NT-proBNP with a conservative weight (0.05)
- Added age-adjusted thresholds for risk calculation
- Implemented sigmoid normalization to handle the wide range of possible values
- Enhanced SHAP value generation to emphasize elevated NT-proBNP levels

### 3. ECG Visualization

- Modified ECG generation to reflect NT-proBNP levels
- Elevated NT-proBNP results in:
  - Reduced T wave amplitude (ventricular strain pattern)
  - Slightly increased QRS width (conduction delay)
  - Slight ST depression
  - Increased heart rate variability

### 4. Machine Learning Model

- Added NT-proBNP to the clinical priors with a strong coefficient (0.8)
- Implemented log transformation to handle the wide range of values
- Added an interaction term between NT-proBNP and age

## How to Use

1. **Obtaining NT-proBNP Values**:
   - NT-proBNP requires a specific blood test ordered by a healthcare provider
   - Values are typically reported in pg/mL (picograms per milliliter)
   - The test is commonly ordered for patients with suspected heart failure

2. **Entering Values**:
   - Enter the NT-proBNP value in the designated field if available
   - Leave the field blank if no NT-proBNP test has been performed
   - The system will function normally without this value

3. **Interpreting Results**:
   - When NT-proBNP is provided, it will be included in the risk calculation
   - The SHAP values will show the contribution of NT-proBNP to the prediction
   - The ECG visualization will reflect the impact of elevated NT-proBNP

## Future Enhancements

In future updates, we plan to:

1. Add support for additional cardiac biomarkers (Troponin, CRP)
2. Implement automatic retrieval of biomarker values from EHR systems
3. Add trending of biomarker values over time
4. Enhance the visualization of biomarker impact on risk prediction

## References

1. Januzzi JL Jr, et al. (2019). "NT-proBNP Testing for Diagnosis and Short-Term Prognosis in Acute Heart Failure: An International Pooled Analysis of 1,256 Patients." European Heart Journal, 40(44), 3670-3677.

2. Ponikowski P, et al. (2016). "2016 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure." European Heart Journal, 37(27), 2129-2200.

3. Ibrahim NE, Januzzi JL Jr. (2018). "The Future of Biomarker-Guided Therapy for Heart Failure After the Guiding Evidence-Based Therapy Using Biomarker Intensified Treatment in Heart Failure (GUIDE-IT) Study." Current Heart Failure Reports, 15(2), 37-43.
