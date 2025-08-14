"""Predict module for predicting outcomes based on input data. 
This module contains functions to load a model and make predictions."""

import pandas as pd
import logging
from load_model import load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict(input_data:pd.DataFrame, model_path:str) -> pd.DataFrame:
    """
    Predict outcomes based on input data using a pre-trained model.

    Args:
        input_data (pd.DataFrame): Input data for prediction.
        model_path (str): Path to the pre-trained model.

    Returns:
        pd.DataFrame: DataFrame containing predictions.
    """

    logging.info("Loading model from %s", model_path)
    # Load the pre-trained model
    model = load_model(model_path)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    logging.info("Predictions made successfully.")
    # Convert predictions to DataFrame
    return pd.DataFrame(predictions, columns=['predicted_outcome']) 

if __name__ == "__main__":
    
    test_data = pd.read_csv('data/test_processed.csv').sample(1000, random_state=42)
    sample_submission = pd.read_csv('data/sample_submission.csv').sample(1000, random_state=42)

    logging.info("Starting prediction process...")

    model_path = 'models/best_model.pkl'
    result = predict(test_data, model_path)
    sample_submission['y'] = result
    sample_submission.to_csv('data/sample_submission1.csv', index=False)
    logging.info("Prediction process completed.")
