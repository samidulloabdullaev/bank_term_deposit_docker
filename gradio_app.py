import gradio as gr
import pandas as pd
import logging
from predict import predict

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = "best_model.pkl"

FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day",
    "month", "duration", "campaign", "pdays", "previous", "poutcome"
]

EXAMPLE_ROW = [
    32, "blue-collar", "married", "secondary", "no",
    1397, "yes", "no", "unknown", 21,
    "may", 224, 1, -1, 0, "unknown"
]

def make_prediction_for_gradio(*features):
    try:
        input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        result_df = predict(input_df, MODEL_PATH, proba=True)
        return f"Predicted Probability: {result_df['predicted_probability'][0]:.4f}"
    except Exception as e:
        logging.exception("Prediction error")
        return f"An error occurred: {str(e)}"

with gr.Blocks() as iface:
    gr.Markdown("## Bank Term Deposit Classification App")
    gr.Markdown("Enter all input features below, then click **Predict** to see the probability.")

    with gr.Row():
        age = gr.Number(label="Age", value=EXAMPLE_ROW[0], precision=0, scale=1)
        job = gr.Textbox(label="Job", value=EXAMPLE_ROW[1], scale=3)
        marital = gr.Textbox(label="Marital Status", value=EXAMPLE_ROW[2], scale=2)
        education = gr.Textbox(label="Education", value=EXAMPLE_ROW[3], scale=2)
        default = gr.Textbox(label="Default", value=EXAMPLE_ROW[4], scale=1)

    with gr.Row():
        balance = gr.Number(label="Balance", value=EXAMPLE_ROW[5], precision=0, scale=2)
        housing = gr.Textbox(label="Housing", value=EXAMPLE_ROW[6], scale=1)
        loan = gr.Textbox(label="Loan", value=EXAMPLE_ROW[7], scale=1)
        contact = gr.Textbox(label="Contact", value=EXAMPLE_ROW[8], scale=2)
        day = gr.Number(label="Day", value=EXAMPLE_ROW[9], precision=0, scale=1)

    with gr.Row():
        month = gr.Textbox(label="Month", value=EXAMPLE_ROW[10], scale=1)
        duration = gr.Number(label="Duration", value=EXAMPLE_ROW[11], precision=0, scale=2)
        campaign = gr.Number(label="Campaign", value=EXAMPLE_ROW[12], precision=0, scale=1)
        pdays = gr.Number(label="Pdays", value=EXAMPLE_ROW[13], precision=0, scale=1)
        previous = gr.Number(label="Previous", value=EXAMPLE_ROW[14], precision=0, scale=1)
        poutcome = gr.Textbox(label="Poutcome", value=EXAMPLE_ROW[15], scale=2)

    submit_btn = gr.Button("Predict", variant="primary")

    output_text = gr.Textbox(label="Model Output", lines=2)

    submit_btn.click(
        fn=make_prediction_for_gradio,
        inputs=[age, job, marital, education, default,
                balance, housing, loan, contact, day,
                month, duration, campaign, pdays, previous, poutcome],
        outputs=output_text
    )

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=8000, share=True)
