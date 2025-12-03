"""
Classifier Service for the Mini ML-based IDS Platform

This service consumes network traffic messages from a Redis queue, performs
real-time intrusion detection using a pre-trained ML model, and publishes
classification results to a Redis pub/sub channel.

The classifier acts as the "brain" of the IDS - it takes incoming traffic
data and determines whether each connection is normal or potentially malicious.

Architecture:
    Redis Queue → Classifier → Redis Pub/Sub Channel → UI/Alerts
"""

import os
import sys
import json
import time
import logging
import redis
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Thread, Event
import signal

# Add parent directory to path to import from src.model
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# CRITICAL: Import preprocessing module BEFORE loading model
# This ensures log2_transform and other functions are available when
# joblib unpickles the preprocessor (which contains FunctionTransformers)
# The model was trained with "from preprocessing import log2_transform"
# so we need to make it available under that name in sys.modules
from src.model import preprocessing  # noqa: F401
sys.modules['preprocessing'] = preprocessing  # Make it available as 'preprocessing'

from src.model.inference import IDSClassifier

# FastAPI for health/metrics endpoints
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("classifier-service")


# =============================================================================
# CONFIGURATION
# =============================================================================

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "traffic_queue")
REDIS_RESULTS_CHANNEL = os.getenv("REDIS_RESULTS_CHANNEL", "classification_results")

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")

# Service configuration
BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", 1))  # seconds to wait for messages
HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", 8000))


# =============================================================================
# METRICS AND STATE
# =============================================================================

class ServiceMetrics:
    """Track service metrics for monitoring and health checks."""

    def __init__(self):
        self.start_time = time.time()
        self.messages_processed = 0
        self.predictions_normal = 0
        self.predictions_attack = 0
        self.errors = 0
        self.last_prediction_time = None
        self.model_loaded = False
        self.redis_connected = False

    def record_prediction(self, prediction: str):
        """Record a prediction."""
        self.messages_processed += 1
        self.last_prediction_time = time.time()

        if prediction == "attack":
            self.predictions_attack += 1
        else:
            self.predictions_normal += 1

    def record_error(self):
        """Record an error."""
        self.errors += 1

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "uptime_seconds": self.get_uptime(),
            "messages_processed": self.messages_processed,
            "predictions": {
                "normal": self.predictions_normal,
                "attack": self.predictions_attack,
                "attack_rate": (
                    self.predictions_attack / self.messages_processed
                    if self.messages_processed > 0
                    else 0
                )
            },
            "errors": self.errors,
            "last_prediction_time": self.last_prediction_time,
            "status": {
                "model_loaded": self.model_loaded,
                "redis_connected": self.redis_connected
            }
        }


# Global metrics instance
metrics = ServiceMetrics()


# =============================================================================
# REDIS CONNECTION
# =============================================================================

def create_redis_connection() -> redis.Redis:
    """
    Create and test a connection to Redis.

    We use a retry loop because when running in Docker, Redis might not be
    ready immediately when our service starts.
    """
    max_retries = 10
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True
            )
            # Test the connection
            client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            metrics.redis_connected = True
            return client
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Could not connect to Redis after {max_retries} attempts")


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model() -> IDSClassifier:
    """
    Load the trained IDS classifier from disk.

    The model artifacts should be mounted as a Docker volume at MODEL_PATH.
    """
    logger.info(f"Loading model from {MODEL_PATH}...")

    try:
        classifier = IDSClassifier(models_dir=MODEL_PATH)
        metrics.model_loaded = True
        logger.info("Model loaded successfully!")
        return classifier
    except FileNotFoundError as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Make sure model artifacts are available at MODEL_PATH")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        raise


# =============================================================================
# MESSAGE PROCESSING
# =============================================================================

def process_message(
    classifier: IDSClassifier,
    message: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single traffic message and return classification results.

    Args:
        classifier: The loaded IDS classifier
        message: Message from Redis queue containing traffic features

    Returns:
        Dictionary with classification results in the format specified by CLAUDE.md
    """
    try:
        # Extract features from message
        features = message.get("features", {})

        if not features:
            logger.warning(f"Message {message.get('id')} has no features")
            return None

        # Perform inference
        prediction_result = classifier.predict_one(features)

        # Build result in format specified by CLAUDE.md
        result = {
            "id": message.get("id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "original_log": message,
            "prediction": "suspicious" if prediction_result["prediction"] == "attack" else "normal",
            "score": prediction_result.get("confidence", 0.0),
            "details": {
                "prediction_label": prediction_result["prediction_label"],
                "probabilities": prediction_result.get("probabilities", {})
            }
        }

        # Record metrics
        metrics.record_prediction(prediction_result["prediction"])

        return result

    except Exception as e:
        logger.error(f"Error processing message {message.get('id')}: {e}")
        metrics.record_error()
        return None


def publish_result(redis_client: redis.Redis, result: Dict[str, Any]):
    """
    Publish classification result to Redis pub/sub channel.

    Args:
        redis_client: Connected Redis client
        result: Classification result to publish
    """
    try:
        redis_client.publish(
            REDIS_RESULTS_CHANNEL,
            json.dumps(result)
        )
    except Exception as e:
        logger.error(f"Error publishing result: {e}")
        metrics.record_error()


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def run_classifier(
    redis_client: redis.Redis,
    classifier: IDSClassifier,
    shutdown_event: Event
):
    """
    Main processing loop that consumes messages from Redis queue and classifies them.

    Args:
        redis_client: Connected Redis client
        classifier: Loaded IDS classifier
        shutdown_event: Event to signal shutdown
    """
    logger.info(f"Starting classifier loop...")
    logger.info(f"  - Consuming from queue: {REDIS_QUEUE}")
    logger.info(f"  - Publishing to channel: {REDIS_RESULTS_CHANNEL}")

    while not shutdown_event.is_set():
        try:
            # BLPOP blocks until a message is available or timeout
            # Returns tuple: (queue_name, message) or None if timeout
            result = redis_client.blpop(REDIS_QUEUE, timeout=BATCH_TIMEOUT)

            if result is None:
                # Timeout - no messages available
                continue

            _, message_json = result

            # Parse message
            try:
                message = json.loads(message_json)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse message JSON: {e}")
                metrics.record_error()
                continue

            # Log received message
            message_id = message.get("id", "unknown")
            logger.debug(f"Processing message {message_id}")

            # Process message
            classification_result = process_message(classifier, message)

            if classification_result:
                # Publish result
                publish_result(redis_client, classification_result)

                # Log result
                prediction = classification_result["prediction"]
                score = classification_result["score"]
                logger.info(
                    f"Message {message_id}: {prediction.upper()} "
                    f"(confidence: {score:.4f})"
                )

                # Log metrics periodically
                if metrics.messages_processed % 100 == 0:
                    logger.info(
                        f"Processed {metrics.messages_processed} messages | "
                        f"Attack rate: {metrics.get_metrics()['predictions']['attack_rate']:.2%}"
                    )

        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            metrics.redis_connected = False
            time.sleep(5)  # Wait before retry
            try:
                redis_client.ping()
                metrics.redis_connected = True
            except:
                pass

        except Exception as e:
            logger.error(f"Unexpected error in classifier loop: {e}")
            metrics.record_error()
            time.sleep(1)  # Brief pause before continuing

    logger.info("Classifier loop stopped")


# =============================================================================
# HEALTH CHECK & METRICS API
# =============================================================================

# Create FastAPI app for health checks and metrics
app = FastAPI(title="IDS Classifier Service")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns 200 OK if service is healthy, 503 Service Unavailable otherwise.
    """
    is_healthy = metrics.model_loaded and metrics.redis_connected

    status_code = 200 if is_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "model_loaded": metrics.model_loaded,
            "redis_connected": metrics.redis_connected,
            "uptime_seconds": metrics.get_uptime()
        }
    )


@app.get("/metrics")
async def get_metrics():
    """
    Metrics endpoint.

    Returns detailed service metrics for monitoring.
    """
    return JSONResponse(content=metrics.get_metrics())


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "IDS Classifier",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics"
        }
    }


def run_api_server():
    """Run the FastAPI server for health checks and metrics."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=HEALTH_CHECK_PORT,
        log_level="warning"  # Reduce uvicorn logging noise
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the classifier service.

    This orchestrates:
    1. Loading the ML model
    2. Connecting to Redis
    3. Starting the health check API
    4. Running the classification loop
    """
    logger.info("=" * 60)
    logger.info("Starting IDS Classifier Service")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  - Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"  - Input Queue: {REDIS_QUEUE}")
    logger.info(f"  - Output Channel: {REDIS_RESULTS_CHANNEL}")
    logger.info(f"  - Model Path: {MODEL_PATH}")
    logger.info(f"  - Health Check Port: {HEALTH_CHECK_PORT}")
    logger.info("=" * 60)

    # Create shutdown event for graceful shutdown
    shutdown_event = Event()

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Step 1: Load model
        classifier = load_model()

        # Step 2: Connect to Redis
        redis_client = create_redis_connection()

        # Step 3: Start health check API in background thread
        logger.info(f"Starting health check API on port {HEALTH_CHECK_PORT}...")
        api_thread = Thread(target=run_api_server, daemon=True)
        api_thread.start()

        # Brief pause to let API start
        time.sleep(1)
        logger.info("Health check API started")

        # Step 4: Run classifier loop
        run_classifier(redis_client, classifier, shutdown_event)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        logger.info("Classifier service stopped.")
        logger.info(f"Final metrics: {metrics.get_metrics()}")


if __name__ == "__main__":
    main()
