import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open("model.bin", "rb") as f_in:
    saved = pickle.load(f_in)

model = saved["model"]
expected_columns = saved["columns"]

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    patient = request.get_json()
    if not isinstance(patient, dict):
        return jsonify({"error": "JSON body must be an object/dict of feature_name: value"}), 400

    X = pd.DataFrame([patient])

    # Reindex to the expected columns to preserve order; if any column is missing, return an informative error.
    missing = [c for c in expected_columns if c not in X.columns]
    if missing:
        return jsonify({
            "error": "Missing required features",
            "missing_features": missing,
            "expected_features": expected_columns
        }), 400

    X = X.reindex(columns=expected_columns)

    # model is a pipeline (scaler + model), so it will handle scaling internally
    prob = model.predict_proba(X)[0, 1]
    result = {
        "probability": float(prob),
        "alzheimers_likely": bool(prob >= 0.5)
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
