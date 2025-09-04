import re

with open('app.py', 'r') as f:
    content = f.read()

# Extract the check_patient route and function
check_patient_match = re.search(r'@app\.route\(\'/api/patients/check/<patient_id>\', methods=\[.*?\]\)\s+def check_patient\(patient_id\):.*?return jsonify\({.*?}\), 500', content, re.DOTALL)
if check_patient_match:
    check_patient_code = check_patient_match.group(0)
    
    # Remove the check_patient route and function
    content = content.replace(check_patient_code, '')
    
    # Add the check_patient route and function after the get_patient route
    get_patient_match = re.search(r'@app\.route\(\'/api/patients/<patient_id>\', methods=\[.*?\]\)\s+def get_patient\(patient_id\):.*?return jsonify\({\'error\': str\(e\)}\), 500', content, re.DOTALL)
    if get_patient_match:
        get_patient_code = get_patient_match.group(0)
        content = content.replace(get_patient_code, get_patient_code + '\n\n' + check_patient_code)
        
        with open('app.py', 'w') as f:
            f.write(content)
        print('Successfully reordered routes in app.py')
    else:
        print('Could not find get_patient route')
else:
    print('Could not find check_patient route')
