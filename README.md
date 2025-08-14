# Machine Learning Application with Docker

This repository contains a structured machine learning project for a bank application. The project includes a complete pipeline from data preprocessing to model training and deployment via both a FastAPI backend and a Gradio user interface. The entire application is containerized using Docker for consistent and reproducible environments.

-----

### 📂 Repository Structure

The project is organized to separate different components of the machine learning lifecycle, making it easy to navigate and maintain.

```
project/
│
├── src/
│   └── bank_ml_app/                 # Main Python package
│       ├── __init__.py              # Makes this a Python package
│       │
│       ├── data/                     # Optional: can keep raw/processed data (not included in build)
│       │   ├── train.csv             # Raw train dataset
│       │   ├── test.csv              # Raw test dataset
│       │   ├── sample_submission.csv # Sample submission file
│       │   ├── train_processed.csv   # Processed train dataset (generated)
│       │   └── test_processed.csv    # Processed test dataset (generated)
│       │
│       ├── preprocessing/           # Feature engineering and preprocessing modules
│       │   ├── __init__.py
│       │   ├── engineering.py        # Feature engineering functions
│       │   ├── processing.py         # Data preprocessing functions
│       │   └── main_preprocess.py    # Main script to execute preprocessing
│       │
│       ├── models/                  # Model training & inference
│       │   ├── __init__.py
│       │   ├── train.py              # Script for training LightGBM
│       │   ├── load_model.py         # Utility for loading saved models
│       │   ├── predict.py            # Functions for making predictions
│       │   └── best_model.pkl        # Trained model (generated)
│       │
│       ├── ui/                      # Gradio web application interface
│       │   ├── __init__.py
│       │   └── gradio_app.py
│       │
│       ├── api/                     # FastAPI entry point
│       │   ├── __init__.py
│       │   └── main.py
│       │
│       └── utils/                   # Miscellaneous helper functions
│           ├── __init__.py
│           └── helpers.py
│
├── tests/                            # Optional: unit tests / integration tests
│   └── test_preprocessing.py
│
├── pyproject.toml                     # Python package build config
├── requirements.txt                   # Optional: for non-build installs
└── Dockerfile                         # Docker image definition
```

-----

### 🚀 Getting Started

To get the application up and running, you'll need **Docker** installed on your system.

#### 1\. Preprocess and Train the Model

Before building the Docker image, you need to preprocess the data and train the model locally.

```bash
# First, run the preprocessing script to generate processed data files
python preprocessing/main_preprocess.py

# Next, run the training script to train the model and save it
python models/train.py
```

This will create `data/train_processed.csv`, `data/test_processed.csv`, and `models/best_model.pkl` which are required for the application to function.

#### 2\. Run with Docker

Once the model is trained, you can build and run the Docker container.

```bash
# Build the Docker image
docker build -t bank-ml-app .

# Run the container for the Gradio app
docker run -p 8000:8000 bank-ml-app
```

You can then access the Gradio application in your browser at `http://localhost:8000`.

-----

### 📄 Detailed Component Description

  * **`preprocessing/`**: Contains the logic for preparing the raw data.
      * `main_preprocess.py`: This is the main script that orchestrates the entire preprocessing pipeline. When run, it reads the raw data, applies feature engineering and transformations, and saves the processed datasets to the `data/` directory.
      * `engineering.py`: Functions for creating new features.
      * `processing.py`: Functions for data cleaning and transformations.
  * **`models/`**: Scripts for model development.
      * `train.py`: This script trains a LightGBM model using cross-validation. It saves the final model as `best_model.pkl` after training.
      * `predict.py`: Contains a function to load a model and make predictions on new data.
  * **`ui/`**: User-facing application code.
      * `gradio_app.py`: This script defines a user-friendly web interface using **Gradio**, allowing for interactive model predictions.
  * **`api/`**: The backend for programmatic access.
      * `main.py`: The entry point for a **FastAPI** application, which provides a RESTful API for model predictions. This is an alternative to the Gradio UI for integration with other services.
  * **`utils/`**: Shared helper functions used across the project.
      * `helpers.py`: Contains utility functions like memory reduction and data sampling.

-----

### 📦 Dependencies

The project's dependencies are listed in `requirements.txt`.

```plaintext
# requirements.txt
python==3.12.0
pandas==2.3.1
numpy==2.3.0
scikit-learn==1.7.1
matplotlib==3.10.5
seaborn==0.13.0
pytest==8.0.0
pytest-cov==6.2.0
jupyter==1.0.0
ipython==7.0.0
requests==2.22.0
fastapi==0.116.1
uvicorn==0.34.0
gradio==5.42.0
joblib==1.5.1
xgboost==3.0.0
lightgbm==4.6.0
optuna==4.4.0
pydantic==2.8.2
```

-----

### 🐳 Docker Configuration

The `Dockerfile` is configured to create a lean and efficient image by installing dependencies and then copying only the necessary project files.

```dockerfile
# Dockerfile
# Use a slim Python base image for a smaller final image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies first.
# This step is separated to leverage Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific project directories and files as requested.
COPY data ./data
COPY models ./models
COPY preprocessing ./preprocessing
COPY ui ./ui
COPY utils ./utils

# Expose the port that the Gradio app will run on.
EXPOSE 8000

# Define the command to run the Gradio application.
CMD ["python", "ui/gradio_app.py"]
```