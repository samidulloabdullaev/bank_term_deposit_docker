# api.py
"""
This file defines a FastAPI application for serving predictions
from the pre-trained LightGBM model.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the loaded model
MODEL = None

class PredictionRequest(BaseModel):
    """
    Pydantic model for the API request body.
    You need to define all the features your model expects here.
    This example uses a dummy list of features.
    
    NOTE: Replace these dummy features with the actual columns from your
    'train_processed.csv' file.
    """
    features: List[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This is the new, recommended way to handle startup and shutdown events.
    The code before `yield` is executed on startup.
    The code after `yield` is executed on shutdown.
    """
    global MODEL
    model_path = 'models/best_model.pkl'
    try:
        logging.info("Loading model from %s...", model_path)
        MODEL = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error("Model file not found at %s. Please train the model first.", model_path)
        # Note: We don't raise an HTTPException here. The /predict endpoint will handle it.
        MODEL = None
    
    # The `yield` statement indicates the application has started
    yield
    
    # You could put shutdown code here if needed
    logging.info("Application shutdown.")

# Initialize FastAPI app with the lifespan context manager
app = FastAPI(
    title="LightGBM Prediction API",
    description="A simple API to predict outcomes using a pre-trained LightGBM model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make a prediction based on the input features.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again later or train the model."
        )

    try:
        # Convert the list of features to a pandas DataFrame
        input_df = pd.DataFrame([request.features])
        
        # Make a prediction. Note: we use predict_proba for LightGBM
        prediction_proba = MODEL.predict_proba(input_df)[:, 1]
        
        return {"predicted_probability": float(prediction_proba[0])}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )
