from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Retail Churn Prediction API")

# Load artifacts once at startup
model = joblib.load('models/rf_churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

class CustomerData(BaseModel):
    Recency: float
    Frequency: int
    Monetary: float
    Cluster: int

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Convert input to DataFrame for the scaler
    input_df = pd.DataFrame([data.dict()])
    
    # Preprocess using the saved scaler
    scaled_data = scaler.transform(input_df)
    
    # Get prediction and probability
    prediction = int(model.predict(scaled_data)[0])
    probability = float(model.predict_proba(scaled_data)[:, 1][0])
    
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "segment_id": data.Cluster
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)