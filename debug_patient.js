const axios = require('axios');

async function testPatientVerification() {
  try {
    // 1. First, let's check if the patient exists
    const patientId = 'patient_20250419123520';
    console.log(`Checking if patient ${patientId} exists...`);
    
    const checkResponse = await axios.get(`http://localhost:8082/api/patients/check/${patientId}`);
    console.log('Check response:', checkResponse.data);
    
    // 2. Now let's get the full patient data
    console.log(`Getting full patient data for ${patientId}...`);
    const patientResponse = await axios.get(`http://localhost:8082/api/patients/${patientId}`);
    console.log('Patient data:', patientResponse.data);
    
    return {
      checkResponse: checkResponse.data,
      patientData: patientResponse.data
    };
  } catch (error) {
    console.error('Error in test:', error.message);
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    return null;
  }
}

testPatientVerification().then(result => {
  console.log('Test completed');
});
