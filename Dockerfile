# docker container config for bank term deposit application# Dockerfile

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
COPY utils.py .
COPY load_model.py .
COPY models/best_model.pkl ./models/

# Expose port for Gradio app
EXPOSE 8000

# Default command
CMD ["python", "gradio_app.py"]
