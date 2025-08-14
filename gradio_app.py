# gradio_app.py
"""
This file defines a Gradio application for serving predictions
from the pre-trained LightGBM model.
"""

import gradio as gr
import joblib
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable for the model, loaded once on startup
MODEL = None
MODEL_PATH = 'models/best_model.pkl'

try:
    logging.info("Loading model from %s...", MODEL_PATH)
    MODEL = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file not found at %s. Please train the model first.", MODEL_PATH)
    MODEL = None
    
# This function will be called by the Gradio interface
def make_prediction_for_gradio(*features):
    """
    Makes a prediction from the Gradio UI input.
    """
    if MODEL is None:
        return "Error: Model not loaded. Please ensure it's trained."

    try:
        # Convert tuple of features to a pandas DataFrame
        # The column names are not important, but the order must be correct
        input_df = pd.DataFrame([features])
        
        # Make a prediction
        prediction_proba = MODEL.predict_proba(input_df)[:, 1]
        
        return f"Predicted Probability: {prediction_proba[0]:.4f}"
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Define the Gradio interface
iface = gr.Interface(
    fn=make_prediction_for_gradio,
    # NOTE: You must replace these with the actual features from your model
    inputs=[gr.Number(label=f"Feature {i+1}") for i in range(10)], # Dummy inputs for demonstration
    outputs="text",
    title="LightGBM Prediction App",
    description="Enter feature values to get a prediction from a pre-trained model."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)

