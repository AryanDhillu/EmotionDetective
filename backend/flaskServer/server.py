from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your machine learning model using pickle
# Ensure 'sent.pkl' file exists in the correct location
# model = pickle.load(open('sent.pkl', 'rb'))

@app.route('/')
def hello():
    return 'Hello, Flask!'

@app.route("/members", methods=['POST'])
def members():
    data = request.get_json()  # Retrieve JSON data from request

    if data is None or 'expression' not in data:
        return jsonify({"error": "Invalid JSON data or missing 'expression'"}), 400

    # Process the expression (replace with your model logic)
    expression = data['expression']
    result = "Dummy result"  # Replace with actual model prediction

    # Prepare response containing original expression and result
    response_data = {
        "expression": expression,
        "result": result
    }

    return jsonify(response_data), 200

if __name__ == "__main__":
    app.run(debug=True)
