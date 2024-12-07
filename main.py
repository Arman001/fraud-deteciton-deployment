from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained Random Forest model
MODEL_PATH = "model/rf_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    # Retrieve the expected feature names if stored during training
    expected_columns = model.feature_names_in_
except AttributeError:
    raise RuntimeError("Model does not have feature names stored. Ensure you use the correct input format.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Define request schema
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Random Forest Prediction API is live!"}

@app.post("/predict/")
def predict(data: InputData):
    try:
        # Convert input features into a pandas DataFrame with expected column names
        features_df = pd.DataFrame([data.features], columns=expected_columns)
        # Make prediction
        prediction = model.predict(features_df)
        return {"prediction": int(prediction[0])}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
