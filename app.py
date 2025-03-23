from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Serves the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from JSON request
    features = [
        data.get('MedInc'),
        data.get('HouseAge'),
        data.get('AveRooms'),
        data.get('AveBedrms'),
        data.get('Population'),
        data.get('AveOccup'),
        data.get('Latitude'),
        data.get('Longitude')
    ]

    # Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)

    # Predict house price
    prediction = model.predict(features)[0]

    return jsonify({'predicted_price': prediction})

@app.route('/predict_html', methods=['POST'])
def predict_html():
    # Extract data from form
    features = [
        float(request.form['MedInc']),
        float(request.form['HouseAge']),
        float(request.form['AveRooms']),
        float(request.form['AveBedrms']),
        float(request.form['Population']),
        float(request.form['AveOccup']),
        float(request.form['Latitude']),
        float(request.form['Longitude'])
    ]

    # Convert to NumPy array and reshape
    features = np.array(features).reshape(1, -1)

    # Predict house price
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction=round(prediction, 2))  # Show result on page

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

