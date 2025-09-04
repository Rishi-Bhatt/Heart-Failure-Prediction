@app.route('/api/patients/check/<patient_id>', methods=['GET', 'POST'])
def check_patient(patient_id):
    """
    Endpoint to check if a patient record exists
    """
    try:
        patient_file = f'data/patients/{patient_id}.json'
        exists = os.path.exists(patient_file)
        
        return jsonify({
            'exists': exists,
            'patient_id': patient_id,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
