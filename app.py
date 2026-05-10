from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

# LOAD MODEL
model = joblib.load("svm_brs_model_probability.pkl")
scaler = joblib.load("scaler.pkl")

labels = [
    "Low Resilience",
    "Normal Resilience",
    "High Resilience"
]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "SVM BRS API Running"

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.json

        features = np.array(data["features"]).reshape(1, -1)

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)

        probabilities = model.predict_proba(scaled_features)

        predicted_class = int(prediction[0])

        confidence = float(np.max(probabilities))

        result = labels[predicted_class]

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
