// Simple script to check the response from the check endpoint
const axios = require('axios');

async function checkPatient(patientId) {
  try {
    console.log(`Checking patient: ${patientId}`);
    const response = await axios.get(`http://localhost:8082/api/patients/check/${patientId}`);
    console.log('Response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error checking patient:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return null;
  }
}

// Check a recent patient
checkPatient('patient_20250419123520');
