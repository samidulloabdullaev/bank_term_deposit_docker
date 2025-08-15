# # docker container config for bank term deposit application# Dockerfile
# # Use a slim Python base image for a smaller final image
# FROM python:3.9-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Copy the requirements file and install dependencies first.
# # This step is separated to leverage Docker's layer caching.
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the specific project directories and files as requested.
# # The project structure is replicated inside the container.
# COPY . .

# # Expose the port that the Gradio app will run on.
# # The default port for Gradio is 7860.
# EXPOSE 8000

# # Define the command to run the Gradio application.
# # The host 0.0.0.0 makes the app accessible from outside the container.
# CMD ["python", "gradio_app.py"]


# Dockerfile - Optimized for smaller size

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
