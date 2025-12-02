# Traffic Generator Service Dockerfile
# =====================================
# This Dockerfile creates a minimal, production-ready image for the
# traffic generator service. We follow best practices:
# - Use a slim base image to reduce size
# - Create a non-root user for security
# - Separate dependency installation from code copying (for better caching)

# Start from Python 3.11 slim image (~150MB vs ~1GB for full)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Create a non-root user for security
# Running as root in containers is a security anti-pattern
RUN useradd --create-home --shell /bin/bash appuser

# Install dependencies first (this layer gets cached)
# We do this before copying the code because dependencies change less often
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables with defaults
# These can be overridden in docker-compose.yml
ENV REDIS_HOST=redis \
    REDIS_PORT=6379 \
    REDIS_QUEUE=traffic_queue \
    TRAFFIC_RATE=10 \
    GENERATOR_MODE=sequential \
    GENERATOR_LOOP=true \
    LOG_LEVEL=INFO

# Run the generator
CMD ["python", "main.py"]
