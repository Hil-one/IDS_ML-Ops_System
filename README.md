# IDS ML-Ops System

A **real-time Machine Learning-based Intrusion Detection System (IDS)** built as a complete MLOps pipeline. This project demonstrates a production-ready streaming ML architecture for network security monitoring.

## Overview

This system simulates a real-world IDS deployment with:
- **Traffic generation** from historical network data
- **Real-time ML inference** for attack detection
- **Live monitoring dashboard** with alerts
- **Full containerization** with Docker
- **Message-driven architecture** using Redis

### Architecture

```
┌─────────────────┐       ┌──────────────┐       ┌─────────────────┐
│     Traffic     │ ───▶  │    Redis     │ ───▶  │   Classifier    │
│    Generator    │       │  (Queue)     │       │    Service      │
└─────────────────┘       └──────────────┘       └─────────────────┘
                                                          │
                                                          ▼
                          ┌──────────────┐       ┌─────────────────┐
                          │    Redis     │ ◀──── │   Trained ML    │
                          │  (Pub/Sub)   │       │     Model       │
                          └──────────────┘       └─────────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │  UI Backend  │
                          │  (WebSocket) │
                          └──────────────┘
                                  │
                                  ▼
                          ┌──────────────┐
                          │ React Dashboard│
                          │     (UI)      │
                          └──────────────┘
```

## Features

### 🎯 **Core Components**

1. **Traffic Generator** (`src/services/generator/`)
   - Streams network traffic from IDS dataset
   - Configurable rate and mode (sequential/random)
   - Publishes to Redis queue

2. **ML Classifier Service** (`src/services/classifier/`)
   - Loads pre-trained Decision Tree model
   - Real-time inference on incoming traffic
   - Publishes results to Redis pub/sub
   - Health and metrics endpoints

3. **Model Training** (`src/model/`)
   - Complete training pipeline
   - Feature engineering and preprocessing
   - Model evaluation and artifact generation
   - Joblib serialization with proper handling

4. **Web Dashboard** (`src/ui/`)
   - Real-time traffic monitoring
   - Live attack detection alerts
   - Statistics and visualizations
   - WebSocket-based updates

### ✨ **Key Features**

- ⚡ **Real-time Processing**: Sub-second latency from traffic to alert
- 🎨 **Modern UI**: React dashboard with live updates
- 🐳 **Fully Dockerized**: One-command deployment
- 📊 **Comprehensive Metrics**: Attack rates, protocol distribution, throughput
- 🧪 **Tested**: Unit tests for core services
- 📚 **Well Documented**: READMEs for every component

## Quick Start

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** v2
- **Python** 3.11+ (for local development)
- **Node.js** 18+ (for UI development)
- **Kaggle API credentials** (for dataset download)

### 1. Clone the Repository

```bash
git clone https://github.com/Hil-one/IDS_ML-Ops_System.git
cd IDS_ML-Ops_System
```

### 2. Train the Model

Before running the system, you need to train the ML model:

```bash
# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (downloads dataset from Kaggle)
python src/model/train.py

# Verify model artifacts
python src/model/verify_artifacts.py
```

This creates model artifacts in `models/` directory:
- `model.joblib` - Trained classifier
- `preprocessor.joblib` - Feature preprocessing pipeline
- `drop_columns.joblib` - Columns to drop
- `model_metadata.joblib` - Model metadata

### 3. Start the System

```bash
# Start all services with Docker Compose
docker-compose up --build
```

This starts:
- **Redis** on port 6379
- **Generator** (background)
- **Classifier** on port 8000
- **UI Backend** on port 5000
- **UI Frontend** on port 3000

### 4. Access the Dashboard

Open your browser to:
- **Dashboard**: http://localhost:3000
- **Classifier Health**: http://localhost:8000/health
- **Backend Health**: http://localhost:5000/health

You should see:
- Real-time traffic logs streaming
- Green/red indicators for normal/attack traffic
- Live statistics and charts
- Alert banners when attacks are detected

## Project Structure

```
IDS_ML-Ops_System/
│
├── src/
│   ├── model/                      # ML model training & inference
│   │   ├── train.py               # Training pipeline
│   │   ├── inference.py           # Inference interface
│   │   ├── preprocessing.py       # Shared preprocessing functions
│   │   ├── evaluate.py            # Model evaluation
│   │   ├── verify_artifacts.py   # Artifact verification
│   │   └── README.md              # Model documentation
│   │
│   ├── services/
│   │   ├── generator/             # Traffic simulation service
│   │   │   ├── main.py
│   │   │   ├── Dockerfile
│   │   │   ├── requirements.txt
│   │   │   └── README.md
│   │   │
│   │   └── classifier/            # ML inference service
│   │       ├── main.py
│   │       ├── Dockerfile
│   │       ├── requirements.txt
│   │       └── README.md
│   │
│   └── ui/                        # Web dashboard
│       ├── src/                   # React components
│       ├── backend/               # WebSocket bridge
│       ├── Dockerfile             # Frontend build
│       ├── Dockerfile.backend     # Backend service
│       └── README.md
│
├── tests/
│   ├── test_generator.py         # Generator unit tests
│   └── test_classifier.py        # Classifier unit tests
│
├── models/                        # Model artifacts (gitignored)
│   ├── model.joblib
│   ├── preprocessor.joblib
│   ├── drop_columns.joblib
│   └── model_metadata.joblib
│
├── docs/                          # Documentation
│   └── JOBLIB_SERIALIZATION_FIX.md
│
├── docker-compose.yml             # Service orchestration
├── requirements.txt               # Python dependencies
├── CLAUDE.md                      # Development guide
└── README.md                      # This file
```

## Usage

### Running Individual Services

#### Generate Traffic Only

```bash
docker-compose up redis generator
```

#### Classifier Only

```bash
# Requires Redis and trained model
docker-compose up redis classifier
```

#### UI Only

```bash
# Requires classifier running
docker-compose up ui-backend ui-frontend
```

### Configuration

Edit `docker-compose.yml` to adjust settings:

**Generator:**
```yaml
environment:
  - TRAFFIC_RATE=10        # Messages per second
  - GENERATOR_MODE=sequential  # or 'random'
  - GENERATOR_LOOP=true    # Loop forever
```

**Classifier:**
```yaml
environment:
  - MODEL_PATH=/app/models
  - BATCH_TIMEOUT=1
  - LOG_LEVEL=INFO
```

### Viewing Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f classifier
docker-compose logs -f generator
docker-compose logs -f ui-backend
```

### Stopping the System

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Development

### Local Development Setup

#### Model Development

```bash
cd src/model
pip install -r requirements.txt

# Train model
python train.py

# Test inference
python inference.py

# Evaluate
python evaluate.py
```

#### Generator Development

```bash
cd src/services/generator
pip install -r requirements.txt
python main.py
```

#### Classifier Development

```bash
cd src/services/classifier
pip install -r requirements.txt
python main.py
```

#### UI Development

```bash
# Backend
cd src/ui
pip install -r backend/requirements.txt
python backend/server.py

# Frontend (separate terminal)
cd src/ui
npm install
npm run dev
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_classifier.py -v
```

## Model Performance

### Dataset

- **Source**: [Kaggle - Intrusion Detection Classifier](https://www.kaggle.com/datasets/ayushparwal2026/intrusion-detection-classifier)
- **Size**: ~125,000 network connection records
- **Features**: 41 features (numerical and categorical)
- **Target**: Binary classification (normal vs attack)

### Model Metrics

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.99  |
| Precision | ~0.99  |
| Recall    | ~0.99  |
| F1 Score  | ~0.99  |

### Model Details

- **Algorithm**: Decision Tree Classifier
- **Max Depth**: 6
- **Criterion**: Entropy
- **Training Time**: ~1-2 minutes
- **Inference Time**: <1ms per prediction

## System Performance

### Throughput

- **Generator**: Configurable (default: 10 msg/s)
- **Classifier**: ~500-1000 predictions/second
- **End-to-end Latency**: <100ms (traffic → UI)

### Resource Usage

| Service | CPU | Memory | Disk |
|---------|-----|--------|------|
| Redis | Low | ~50MB | Minimal |
| Generator | Low | ~200MB | Minimal |
| Classifier | Medium | ~300MB | ~500KB (model) |
| UI Backend | Low | ~100MB | Minimal |
| UI Frontend | Low | ~50MB | ~2MB (static) |

## Troubleshooting

### Model Not Found

**Error**: `FileNotFoundError: Model file not found`

**Solution**: Train the model first:
```bash
python src/model/train.py
python src/model/verify_artifacts.py
```

### Classifier Not Starting

**Symptoms**: Container exits immediately

**Solutions**:
1. Check model artifacts are present: `ls -l models/`
2. Check logs: `docker-compose logs classifier`
3. Verify Redis is running: `docker-compose ps redis`

### UI Not Connecting

**Symptoms**: "Disconnected" in dashboard

**Solutions**:
1. Check backend is running: `curl http://localhost:5000/health`
2. Check classifier is publishing: `docker-compose logs classifier`
3. Verify Redis pub/sub: `redis-cli SUBSCRIBE classification_results`

### No Traffic Appearing

**Symptoms**: Empty log stream in UI

**Solutions**:
1. Check generator is running: `docker-compose logs generator`
2. Verify Redis queue: `redis-cli LLEN traffic_queue`
3. Check classifier is consuming: Look for "Processing message" in logs

## Security Considerations

⚠️ **Important for Production Deployments:**

1. **Change Default Secrets**: Update `SECRET_KEY` in docker-compose.yml
2. **Enable HTTPS**: Use reverse proxy (nginx, Traefik) with SSL
3. **Add Authentication**: Implement user auth for UI access
4. **Network Isolation**: Use private networks, expose only necessary ports
5. **Resource Limits**: Add memory and CPU limits to services
6. **Monitoring**: Implement logging aggregation and alerting
7. **Model Versioning**: Use MLflow or similar for model management

## Future Enhancements

- [ ] Multi-class classification (identify specific attack types)
- [ ] Model retraining pipeline with new data
- [ ] Grafana dashboards for metrics
- [ ] Prometheus integration
- [ ] Alert notifications (email, Slack, PagerDuty)
- [ ] Historical data storage and analysis
- [ ] A/B testing for model versions
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline with GitHub Actions
- [ ] API for external integrations

## Contributing

This is an educational/demonstration project. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Technology Stack

### Backend
- **Python 3.11**: Core language
- **FastAPI**: REST endpoints
- **Flask-SocketIO**: WebSocket support
- **Redis**: Message broker
- **Scikit-learn**: ML framework
- **Pandas**: Data processing

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool
- **Socket.IO Client**: Real-time communication
- **Recharts**: Data visualization

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Orchestration
- **Nginx**: Web server
- **Redis**: Queue and pub/sub

## License

This project is provided as-is for educational purposes.

## Acknowledgments

- Dataset: [Kaggle IDS Dataset](https://www.kaggle.com/datasets/ayushparwal2026/intrusion-detection-classifier)
- Developed with assistance from Claude Code

## Contact & Support

For questions or issues:
- **GitHub Issues**: [Create an issue](https://github.com/Hil-one/IDS_ML-Ops_System/issues)
- **Project Repository**: https://github.com/Hil-one/IDS_ML-Ops_System

---

**Built with ❤️ as an MLOps demonstration project**
