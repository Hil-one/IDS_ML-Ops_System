# Traffic Generator Service

The Traffic Generator is a core component of the Mini ML-based IDS Platform. It simulates real-time network traffic by reading records from the test dataset and publishing them to a Redis queue.

## What It Does

Think of the generator as a "replay machine" for network traffic. It takes historical network connection records from a known IDS dataset and streams them as if they were happening live. This allows us to:

1. **Test the classifier** with realistic data without needing actual network capture tools
2. **Demonstrate the system** to stakeholders with controlled, reproducible scenarios
3. **Benchmark performance** under different traffic rates
4. **Validate detection accuracy** since we know the ground truth labels

## How It Works

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   IDS Dataset   │ ───▶  │    Generator    │ ───▶  │   Redis Queue   │
│  (Test Split)   │       │   (This Code)   │       │  traffic_queue  │
└─────────────────┘       └─────────────────┘       └─────────────────┘
                                                            │
                                                            ▼
                                                    ┌─────────────────┐
                                                    │   Classifier    │
                                                    │    Service      │
                                                    └─────────────────┘
```

1. On startup, the generator downloads the IDS dataset (or loads from local file)
2. It extracts the test split (same 15% used to evaluate the model)
3. For each record, it creates a JSON message with all features
4. Messages are pushed to the Redis queue at a configurable rate
5. The process loops forever (or stops after one pass, depending on config)

## Message Format

Each message published to Redis has this structure:

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
  "ground_truth": {
    "target": "normal.",
    "attack_type": "normal",
    "is_attack": 0
  },
  "raw_log": "0,tcp,http,SF,181,5450,..."
}
```

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server hostname |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_QUEUE` | `traffic_queue` | Name of the Redis list to publish to |
| `TRAFFIC_RATE` | `10` | Messages per second |
| `GENERATOR_MODE` | `sequential` | `sequential` or `random` |
| `GENERATOR_LOOP` | `true` | Whether to loop through dataset forever |
| `DATA_PATH` | (none) | Local CSV file path (skips Kaggle download) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Running Locally

### Prerequisites
- Python 3.11+
- Redis server running locally
- Kaggle API credentials (for dataset download)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Redis (if not already running)
redis-server

# Run the generator
python main.py
```

### With Docker
```bash
# From the project root directory
docker-compose up generator
```

## Running Tests

```bash
# From the project root
pytest tests/test_generator.py -v
```

## Notes

- The generator uses the same train/test split as the training notebook to ensure we're testing on unseen data
- Ground truth labels are included for evaluation purposes only - the classifier should not use them!
- The `random` mode shuffles the dataset each pass, useful for simulating varied traffic
- If Redis is unavailable at startup, the generator retries 10 times before giving up
