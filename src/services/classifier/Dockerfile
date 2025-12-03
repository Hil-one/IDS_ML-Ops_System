# Classifier Service Dockerfile
# ==============================
# This Dockerfile creates a minimal, production-ready image for the
# classifier service. The service performs real-time intrusion detection
# by consuming traffic from Redis and running ML inference.

# Start from Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Install dependencies first (for better layer caching)
COPY src/services/classifier/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the classifier service code
COPY src/services/classifier/main.py .

# Copy the model inference module from src/model
# This is needed because the classifier imports from src.model.inference
RUN mkdir -p /app/src/model
COPY src/model/__init__.py /app/src/model/
COPY src/model/inference.py /app/src/model/
COPY src/model/preprocessing.py /app/src/model/

# Create models directory for mounting trained model artifacts
RUN mkdir -p /app/models && chown -R appuser:appuser /app/models

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables with defaults
# These can be overridden in docker-compose.yml
ENV REDIS_HOST=redis \
    REDIS_PORT=6379 \
    REDIS_QUEUE=traffic_queue \
    REDIS_RESULTS_CHANNEL=classification_results \
    MODEL_PATH=/app/models \
    BATCH_TIMEOUT=1 \
    HEALTH_CHECK_PORT=8000 \
    LOG_LEVEL=INFO

# Expose the health check port
EXPOSE 8000

# Run the classifier service
CMD ["python", "main.py"]
