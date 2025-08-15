"""Predict module for predicting outcomes based on input data."""

import pandas as pd
import logging
from load_model import load_model
from preprocess import convert_object_to_category

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict(input_data: pd.DataFrame, model_path: str, proba: bool = False) -> pd.DataFrame:
    """
    Predict outcomes or probabilities based on input data using a pre-trained model.
    """
    logging.info("Loading model from %s", model_path)
    model = load_model(model_path)
    if model is None:
        raise ValueError(f"Model could not be loaded from {model_path}")

    input_data = convert_object_to_category(input_data)

    if proba and hasattr(model, "predict_proba"):
        predictions = model.predict_proba(input_data)[:, 1]
        return pd.DataFrame(predictions, columns=['predicted_probability'])
    else:
        predictions = model.predict(input_data)
        return pd.DataFrame(predictions, columns=['predicted_outcome'])
