# machine-learning-zoomcamp project

# Problem statment: 
Many people will develop Alzheimer's disease. Current medications work better when the
illness is in it's early stages. Want to find the best model which will predict who is most like to develop
the disease, so these people can get treated earlier.

Diagnosis is the target variable.

# Dataset description: 
Here is a link to the raw dataset
https://raw.githubusercontent.com/Mik-Nowak-05/Alzheimers_Disease_Prediction_Model/refs/heads/main/alzheimers_disease_data.csv

# EDA summary: 

These are the header columns of the raw data:
PatientID,Age,Gender,Ethnicity,EducationLevel,BMI,Smoking,AlcoholConsumption,PhysicalActivity,DietQuality,SleepQuality,FamilyHistoryAlzheimers,CardiovascularDisease,Diabetes,Depression,HeadInjury,Hypertension,SystolicBP,DiastolicBP,CholesterolTotal,CholesterolLDL,CholesterolHDL,CholesterolTriglycerides,MMSE,FunctionalAssessment,MemoryComplaints,BehavioralProblems,ADL,Confusion,Disorientation,PersonalityChanges,DifficultyCompletingTasks,Forgetfulness,Diagnosis,DoctorInCharge

The data in the last two columns are removed. DoctorInCharge is not part of the patient's symptoms. The Diagnosis is the target variable. 
The prediction model returns 1 if the patient will have Alzheimer's, returns 0 if not.

# Modeling approach & metrics
Four models were tested: 
LogisticRegression max_iter=500, "C": [0.1, 1, 10] 
DecisionTree "max_depth": [3, 5, 10], "min_samples_leaf": [1, 5, 10]
RandomForest random_state=42,"n_estimators": [100, 300], "max_depth": [None, 10, 20]
GradientBoosting random_state=42,"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]

RandomForest scored the highest.

# How to run locally 
docker build -t alzheimers-api-test .

## Known limitations / next steps
In the file serve.py, I hardcoded patient id and data. A better approach will be read in this data from a file.

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







