const fs = require('fs');
const path = require('path');

// Read the file
const filePath = path.join('src', 'components', 'ResultsDisplay.jsx');
let content = fs.readFileSync(filePath, 'utf8');

// Find and replace the problematic code
const oldCode = `      // Get top 10 features by absolute SHAP value
      const shapData = predictionResult.shap_values;
      const featureImportance = shapData.feature_names.map((name, index) => ({
        name: name,
        value: shapData.values[index],
      }));`;

const newCode = `      // Get top 10 features by absolute SHAP value
      const shapData = predictionResult.shap_values;
      
      // Handle nested array structure of SHAP values
      let shapValues = shapData.values;
      if (Array.isArray(shapValues) && shapValues.length > 0 && Array.isArray(shapValues[0])) {
        shapValues = shapValues[0];
      }
      
      const featureImportance = shapData.feature_names.map((name, index) => ({
        name: name,
        value: shapValues[index] || 0,
      }));`;

content = content.replace(oldCode, newCode);

// Write the updated content back to the file
fs.writeFileSync(filePath, content, 'utf8');
console.log('Successfully updated ResultsDisplay.jsx to fix SHAP values handling');
