FROM python:3.9-slim

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models/production data

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Expose ports for FastAPI and MLflow
EXPOSE 8000
EXPOSE 5000

# Create a script to run both services
RUN echo '#!/bin/bash\n\
    mlflow ui --host 0.0.0.0 --port 5000 &\n\
    cd src/api && uvicorn main:app --host 0.0.0.0 --port 8000\n\
    ' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"] 