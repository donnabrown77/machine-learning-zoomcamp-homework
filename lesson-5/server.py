from fastapi import FastAPI

app = FastAPI()

@app.post("/")
def create_client(client: dict):
    return {"received": client}
