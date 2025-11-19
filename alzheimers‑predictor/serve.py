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

# patient_id = 4753

# patient = {
#     "Age": 73,
#     "Gender": 0,
#     "Ethnicity": 3,
#     "EducationLevel": 1,
#     "BMI": 17.795882442817113,
#     "Smoking": 0,
#     "AlcoholConsumption": 19.55508452555359,
#     "PhysicalActivity": 7.844987790974517,
#     "DietQuality": 1.826334664579784,
#     "SleepQuality": 9.673574157961111,
#     "FamilyHistoryAlzheimers": 1,
#     "CardiovascularDisease": 0,
#     "Diabetes": 0,
#     "Depression": 0,
#     "HeadInjury": 0,
#     "Hypertension": 0,
#     "SystolicBP": 99,
#     "DiastolicBP": 116,
#     "CholesterolTotal": 284.1818577646338,
#     "CholesterolLDL": 153.3227621844376,
#     "CholesterolHDL": 69.77229186479597,
#     "CholesterolTriglycerides": 83.63832413899468,
#     "MMSE": 7.356248624670334,
#     "FunctionalAssessment": 5.895077345354194,
#     "MemoryComplaints": 0,
#     "BehavioralProblems": 0,
#     "ADL": 7.119547742738579,
#     "Confusion": 0,
#     "Disorientation": 1,
#     "PersonalityChanges": 0,
#     "DifficultyCompletingTasks": 1,
#     "Forgetfulness": 0
# }

response = requests.post(url, json=patient)
result = response.json()
print(result)

alzheimers_likely = result["alzheimers_likely"]
print(alzheimers_likely)   # True or False
