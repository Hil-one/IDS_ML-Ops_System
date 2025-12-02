# Classifier Service

The Classifier Service is the "brain" of the Mini ML-based IDS Platform. It performs real-time network intrusion detection by consuming traffic messages from a Redis queue, running ML inference, and publishing classification results.

## What It Does

This service bridges the gap between raw network traffic and actionable security insights:

1. **Consumes Messages** from the Redis queue populated by the traffic generator
2. **Loads a Pre-trained Model** (trained offline) at startup
3. **Performs Real-time Inference** on each network connection
4. **Publishes Results** to a Redis pub/sub channel for the UI to consume
5. **Exposes Metrics** via HTTP endpoints for monitoring

## How It Works

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Redis Queue   │ ───▶  │   Classifier    │ ───▶  │  Redis Pub/Sub  │
│ traffic_queue   │       │   (This Code)   │       │     Channel     │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                  │                          │
                                  ▼                          ▼
                          ┌─────────────────┐       ┌─────────────────┐
                          │  Trained Model  │       │    UI/Alerts    │
                          │  (ML Artifacts) │       │                 │
                          └─────────────────┘       └─────────────────┘
```

### Processing Flow

1. **Startup:**
   - Load model artifacts from mounted volume
   - Connect to Redis
   - Start health check API server

2. **Main Loop:**
   - Block waiting for messages on Redis queue (`BLPOP`)
   - Parse message and extract features
   - Run inference using pre-trained model
   - Publish classification result to pub/sub channel
   - Update metrics

3. **Health & Metrics:**
   - HTTP endpoints available on port 8000
   - `/health` - Service health status
   - `/metrics` - Detailed performance metrics

## Message Format

### Input (from traffic_queue)

The classifier expects messages in the format produced by the generator service:

```json
{
  "id": 1,
  "timestamp": "2024-12-01T14:30:00.123456+00:00",
  "features": {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 181,
    "dst_bytes": 5450,
    ...
  },
  "ground_truth": {...},
  "raw_log": "..."
}
```

### Output (to classification_results channel)

The classifier publishes results in this format (per CLAUDE.md spec):

```json
{
  "id": 1,
  "timestamp": "2024-12-01T14:30:00.789012+00:00",
  "original_log": {...},
  "prediction": "suspicious" | "normal",
  "score": 0.85,
  "details": {
    "prediction_label": 1,
    "probabilities": {
      "normal": 0.15,
      "attack": 0.85
    }
  }
}
```

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_QUEUE` | `traffic_queue` | Input queue name |
| `REDIS_RESULTS_CHANNEL` | `classification_results` | Output pub/sub channel |
| `MODEL_PATH` | `/app/models` | Path to model artifacts directory |
| `BATCH_TIMEOUT` | `1` | Seconds to wait for messages before timeout |
| `HEALTH_CHECK_PORT` | `8000` | Port for health/metrics API |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Model Artifacts

The classifier requires the following model artifacts (trained offline):

```
models/
├── model.joblib              # Trained classifier
├── preprocessor.joblib       # Feature preprocessing pipeline
├── drop_columns.joblib       # Columns to drop after preprocessing
└── model_metadata.joblib     # Model metadata and metrics
```

These must be mounted as a Docker volume at `MODEL_PATH`.

## Running Locally

### Prerequisites
- Python 3.11+
- Redis server running locally
- Trained model artifacts in `models/` directory

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model artifacts exist
ls -l ../../../models/
# Should show: model.joblib, preprocessor.joblib, drop_columns.joblib, model_metadata.joblib

# Start Redis (if not already running)
redis-server

# Run the classifier
python main.py
```

### With Docker

```bash
# From the project root directory
docker-compose up classifier
```

## Health Check & Metrics

The classifier exposes HTTP endpoints for monitoring:

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "redis_connected": true,
  "uptime_seconds": 3600.5
}
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "uptime_seconds": 3600.5,
  "messages_processed": 1523,
  "predictions": {
    "normal": 1200,
    "attack": 323,
    "attack_rate": 0.212
  },
  "errors": 0,
  "last_prediction_time": 1701453678.123,
  "status": {
    "model_loaded": true,
    "redis_connected": true
  }
}
```

## Architecture Details

### Threading Model

The service runs two concurrent threads:

1. **Main Thread**: Runs the classification loop (blocking on Redis queue)
2. **API Thread**: Runs the FastAPI server for health/metrics endpoints

### Error Handling

- **Redis Connection Loss**: Auto-retry with exponential backoff
- **Invalid Messages**: Logged and skipped, doesn't crash the service
- **Model Errors**: Logged as errors, increments error counter
- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals cleanly

### Performance Considerations

- Uses `BLPOP` for efficient queue consumption (blocks until message available)
- Single-threaded inference (appropriate for I/O-bound workload)
- Minimal memory footprint (model loaded once at startup)
- FastAPI runs in daemon thread with minimal overhead

## Testing

```bash
# From the project root
pytest tests/test_classifier.py -v
```

## Integration with Other Services

### Generator Service
- **Input**: Consumes messages from `traffic_queue`
- **Format**: Expects JSON with `features` dict

### UI Service
- **Output**: Publishes to `classification_results` channel
- **Format**: JSON with `prediction`, `score`, and metadata

### Model Training
- **Dependency**: Requires model artifacts from training pipeline
- **Location**: Model files must exist at `MODEL_PATH`

## Troubleshooting

### Model Not Found

```
FileNotFoundError: Model file not found at /app/models/model.joblib
```

**Solution**: Ensure model artifacts are mounted correctly:
- Check `docker-compose.yml` volume configuration
- Verify model files exist locally
- Run training pipeline if models don't exist

### Redis Connection Failed

```
RuntimeError: Could not connect to Redis after 10 attempts
```

**Solution**:
- Ensure Redis is running: `docker-compose up redis`
- Check `REDIS_HOST` and `REDIS_PORT` environment variables
- Verify network connectivity between containers

### No Messages Being Processed

**Symptoms**: Service is healthy but `messages_processed` stays at 0

**Solution**:
- Check if generator service is running
- Verify generator is publishing to the same queue name
- Inspect Redis queue: `redis-cli LLEN traffic_queue`

## Performance Benchmarks

Typical performance on a standard development machine:

- **Throughput**: ~500-1000 messages/second (depends on model complexity)
- **Latency**: <10ms per prediction (model inference only)
- **Memory**: ~200-500MB (depends on model size)

## Notes

- The classifier ignores `ground_truth` data from messages (only for evaluation)
- Predictions are independent - no state maintained between messages
- Service is stateless and can be horizontally scaled if needed
- All predictions are logged at INFO level for audit trail
