import pickle
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'random_forest_model.pkl'
 
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
 
app = Flask('Diagnosis')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X = pd.DataFrame([patient])
    y_pred = model.predict_proba(X)[0, 1]
    print(f'Prediction probability: {y_pred}')
    return jsonify({
        "probability": float(y_pred),
        "alzheimers_likely": bool(y_pred >= 0.5)
    })
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
