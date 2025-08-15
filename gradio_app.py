"""
Gradio app for serving predictions from the pre-trained LightGBM model.
"""

import gradio as gr
import pandas as pd
import logging
from predict import predict  # Use the function from predict.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'models/best_model.pkl'

# Define the actual features your model expects (order is important)
FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "duration", "campaign", "pdays", "previous", "poutcome"
]

def make_prediction_for_gradio(*features):
    """
    Takes features from the Gradio UI, creates a DataFrame, and predicts.
    """
    try:
        # Create DataFrame with correct column names
        input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # Run prediction with probability output
        result_df = predict(input_df, MODEL_PATH, proba=True)
        logging.info("Prediction completed successfully.")
        return f"Predicted Probability: {result_df['predicted_probability'][0]:.4f}"
    except Exception as e:
        logging.exception("Prediction error")
        return f"An error occurred: {str(e)}"

# Define Gradio inputs
inputs = [
    gr.Number(label="Age"),
    gr.Textbox(label="Job"),
    gr.Textbox(label="Marital Status"),
    gr.Textbox(label="Education"),
    gr.Textbox(label="Default"),
    gr.Number(label="Balance"),
    gr.Textbox(label="Housing"),
    gr.Textbox(label="Loan"),
    gr.Textbox(label="Contact"),
    gr.Number(label="Day"),
    gr.Textbox(label="Month"),
    gr.Number(label="Duration"),
    gr.Number(label="Campaign"),
    gr.Number(label="Pdays"),
    gr.Number(label="Previous"),
    gr.Textbox(label="Poutcome")
]

iface = gr.Interface(
    fn=make_prediction_for_gradio,
    inputs=inputs,
    outputs="text",
    title="LightGBM Prediction App",
    description="Enter feature values to get a probability prediction from the trained model."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000, share=True)
