# loading model if model is already trained with joblib 
import joblib

def load_model(model_path):
    """
    Load a pre-trained model from the specified path.

    Parameters:
    model_path (str): The file path to the saved model.

    Returns:
    model: The loaded model.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None