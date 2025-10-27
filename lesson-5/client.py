from fastapi import FastAPI
import requests







# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# url = "http://127.0.0.1:8000"
# client = {
#     "lead_source": "organic_search",
#     "number_of_courses_viewed": 4,
#     "annual_income": 80304.0
# }
# requests.post(url, json=client).json()
import requests

url = "http://127.0.0.1:8000"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

response = requests.post(url, json=client)
print(response.json())
