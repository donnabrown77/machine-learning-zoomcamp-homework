import requests


url = "http://localhost:9696/predict"

patient_id = 6898

patient = {
    "Age": 78,
    "Gender": 1,
    "Ethnicity": 3,
    "EducationLevel": 1,
    "BMI": 15.299911224725683,
    "Smoking": 0,
    "AlcoholConsumption": 8.674505178970971,
    "PhysicalActivity": 6.354281752847727,
    "DietQuality": 1.2634274888256836,
    "SleepQuality": 8.322873960983284,
    "FamilyHistoryAlzheimers": 0,
    "CardiovascularDisease": 1,
    "Diabetes": 0,
    "Depression": 0,
    "HeadInjury": 0,
    "Hypertension": 0,
    "SystolicBP": 103,
    "DiastolicBP": 96,
    "CholesterolTotal": 242.19719200578226,
    "CholesterolLDL": 52.482960636295715,
    "CholesterolHDL": 81.28111102002694,
    "CholesterolTriglycerides": 145.25374617827902,
    "MMSE": 4.030490878240885,
    "FunctionalAssessment": 5.173890959228204,
    "MemoryComplaints": 0,
    "BehavioralProblems": 0,
    "ADL": 3.7853987136912446,
    "Confusion": 0,
    "Disorientation": 0,
    "PersonalityChanges": 0,
    "DifficultyCompletingTasks": 0,
    "Forgetfulness": 1
}

response = requests.post(url, json=patient)
result = response

print(result)

if result:
    print(f"{patient_id} likely to develop Alzheimer's")
else:
    print(f"{patient_id} unlikely to develop Alzheimer's")
