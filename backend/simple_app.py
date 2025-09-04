from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'message': 'Backend is working!'
    })

if __name__ == '__main__':
    print("Starting simple Flask server...")
    app.run(debug=True, port=5000)
