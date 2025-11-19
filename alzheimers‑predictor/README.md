# machine-learning-zoomcamp project

Problem: Many people will develop Alzheimer's disease. Current medications work better when the
illness is in it's early stages. Want to find the best model which will predict who is most like to develop
the disease, so these people can get treated earlier.

Diagnosis is the target variable.

# Alzheimer's Disease Prediction Service

A Flask-based ML service that predicts the likelihood of Alzheimer's disease.

## Endpoints

- GET /health → 200 OK
- POST /predict → Returns prediction JSON

## Example curl

```bash
curl http://localhost:9696/health
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":73,"Gender":0,...}'
