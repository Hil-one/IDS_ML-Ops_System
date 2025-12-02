# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It defines the architecture, service boundaries, technologies, and expectations for code generation.

## Project Overview

This is a **Mini ML-based IDS (Intrusion Detection System) Platform** - an MLOps project building a real-time intrusion detection pipeline. The system trains a classifier on IDS datasets, simulates live network traffic through a message queue, performs real-time inference, and displays alerts via a web UI.

**Tech Stack:**
1. Python Services: FastAPI, Pydantic, Redis client, uvicorn.

2. Frontend: React + Vite or CRA.

3. Infrastructure: Docker, Docker Compose v2.

4. Format Everything as Follows:

    - Each service in its own directory.

    - Each Dockerfile minimal and production‑oriented.

    - Code oriented toward readability and maintainability.
```
project/
simulator/
classifier/
api_backend/
frontend/
docker-compose.yml
README.md
claude.md 
```

## Architecture

The system follows a **streaming ML pipeline** architecture:

```
Traffic Generator → Redis Queue → Classifier Service → Results Queue → UI
                                         ↓
                                  Trained Model
                                  (loaded at startup)
```

**Key Components:**
1. **Model Training** (`src/model/train.py`): already done offline from a jupyter notebook
   
2. **Traffic Generator (Python)** (`src/services/generator/`): Reads entries from a dataset test split, sends records to Redis as messages, simulates real‑time logs with adjustable delay, must be containerized. test samples to Redis
   
3. **Classifier Service** (`src/services/classifier/`): Loads a model (trained externally), subscribes to Redis queue, returns classification results, publishes alerts to Redis channel (pub/sub), exposes health and metrics endpoints, must be containerized.
   
4. **UI (REACT.js)** (`src/ui/`): Displays live logs, shows classification status (green/red), shows alert banners, must be containerized.

5. **Infrastructure (Docker Compose)** : Production‑like local orchestration, configuration for Redis, networks, volumes, reproducible environment.

**Data Flow:**
- Raw IDS datasets → `data/raw/`
- Preprocessed data → `data/processed/`

## Testing Expectations

Claude should: Generate testable modular Python code, include example test data, add unit tests for each service, provide integration test examples for docker-compose.

## Development Commands

### Environment Setup
```bash
# Initial setup (Linux/Mac)
./setup.sh

# Windows
setup.bat

# Manual setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_model.py

# Run single test
pytest tests/test_model.py::test_predict_one_sample
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/
```

### Model Training
```bash
# Train baseline model (once implemented)
python src/model/train.py

# Expected outputs:
# - model.joblib (trained classifier)
# - preprocessor.joblib (feature preprocessing pipeline)
# - metrics report (markdown or notebook)
```

### Running Services (Docker)
```bash
# Start entire stack (once docker-compose.yml exists)
docker-compose up

# Start specific services
docker-compose up generator queue
docker-compose up classifier

# Rebuild after code changes
docker-compose up --build
```

## Project Phases

This project follows a phased development plan (see `Project_Plan.txt`):

- **Phase 0** (Current): Project setup, folder structure, tech stack decisions
- **Phase 1**: Data pipeline and baseline model training
- **Phase 2**: Traffic generator and Redis integration
- **Phase 3**: Classifier service consuming queue messages
- **Phase 4**: UI and alerting
- **Phase 5**: Full Dockerization and CI/CD

## Important Patterns

### Message Format
Queue messages should be JSON-serialized network traffic logs:
```json
{
  "timestamp": "2024-11-30T19:00:00Z",
  "features": {...},  // Preprocessed features matching model input
  "raw_log": "..."    // Original log line
}
```

### Classifier Output Format
```json
{
  "original_log": {...},
  "prediction": "suspicious" | "normal",
  "score": 0.85,
  "timestamp": "2024-11-30T19:00:01Z"
}
```

### Model Artifacts
- Always save both model AND preprocessing pipeline together
- Use joblib for serialization (`.joblib` extension)
- Store in a dedicated `models/` directory (gitignored)
- Mount as Docker volume for classifier service

### Environment Configuration
Use `.env` files for configuration (gitignored):
- `REDIS_HOST`, `REDIS_PORT`
- `MODEL_PATH`
- `TRAFFIC_RATE` (messages per second for generator)
- `LOG_LEVEL`

## Key Design Decisions



## Implementation Priority Order

1. Create folder structure + basic docker-compose.yml.

2. Implement Redis queue + test messages.

3. Build Traffic Simulator.

4. Implement Model Classifier (dummy model first, real model later).

5. Implement UI backend.

6. Build simple React UI.

7. Add logging, metrics, health endpoints.

8. Add CI/CD workflows (optional).

## Claude Code Agent Rules (VERY IMPORTANT)

When assisting in implementation, Claude must:

1. Never improvise architecture — always follow what’s defined here.

2. Generate code per service only when asked.

3. Avoid coupling services.

4. Use environment variables for configuration.

5. Keep Dockerfiles minimal.

6. Ask clarifying questions when requirements are ambiguous.

7. Avoid creating unnecessary complexity.