# Machine Learning Application with Docker

This repository contains a structured machine learning project for a bank application. The project includes a complete pipeline from data preprocessing to model training and deployment via both a FastAPI backend and a Gradio user interface. The entire application is containerized using Docker for consistent and reproducible environments.

-----

### 📂 Repository Structure

The project is organized to separate different components of the machine learning lifecycle, making it easy to navigate and maintain.

```
project/
│
├── data/                     # Optional: can keep raw/processed data (not included in build)
│   ├── train.csv             # Raw train dataset
│   ├── test.csv              # Raw test dataset
│   ├── sample_submission.csv # Sample submission file
│   ├── train_processed.csv   # Processed train dataset (generated)
│   └── test_processed.csv    # Processed test dataset (generated)
│
├── preprocessing/           # Feature engineering and preprocessing modules
│   ├── engineering.py        # Feature engineering functions
│   ├── processing.py         # Data preprocessing functions
│   └── main_preprocess.py    # Main script to execute preprocessing
│
├── models/                  # Model training & inference
│   ├── train.py              # Script for training LightGBM
│   ├── load_model.py         # Utility for loading saved models
│   ├── predict.py            # Functions for making predictions
│   └── best_model.pkl        # Trained model (generated)
│
├── ui/                      # Gradio web application interface
│   └── gradio_app.py
│
├── api/                     # FastAPI entry point
│   └── main.py
│
└── utils/                   # Miscellaneous helper functions
│   └── helpers.py
│
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
python main_preprocess.py

# Next, run the training script to train the model and save it
python train.py
```

This will create `data/train_processed.csv`, `data/test_processed.csv`, and `best_model.pkl` which are required for the application to function.

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
pandas==2.3.1
numpy>=2.0.2
scikit-learn>=1.6.1
matplotlib>=3.5.5
requests>=2.12.0
fastapi>=0.106.1
uvicorn>=0.30.0
gradio==4.44.0
joblib>=1.3.1
lightgbm>=4.4.0
pydantic>=2.6.2
```

-----

### 🐳 Docker Configuration

The `Dockerfile` is configured to create a lean and efficient image by installing dependencies and then copying only the necessary project files.

```dockerfile
# Dockerfile
# Base slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for LightGBM and Pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary app files
COPY gradio_app.py .
COPY api.py .
COPY predict.py .
COPY best_model.pkl .

# Expose port for Gradio app
EXPOSE 8000

# Default command
CMD ["python", "gradio_app.py"]
```