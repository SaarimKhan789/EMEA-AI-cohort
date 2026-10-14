from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import numpy as np

app = FastAPI()

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
async def predict(
    age: int = Form(...),
    income: float = Form(...)
):
    
    if age <= 0:
        return {"prediction": "Age must be greater than 0"}

    if income <= 0:
        return {"prediction": "Income must be greater than 0"}

    features = np.array([[age, income]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        result = "Will Purchase Insurance"
    else:
        result = "Will Not Purchase Insurance"

    return {"prediction": result}
