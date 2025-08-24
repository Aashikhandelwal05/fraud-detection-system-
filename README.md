# Real-time Financial Fraud Detection System with MLOps Pipeline

A comprehensive end-to-end machine learning system for detecting fraudulent credit card transactions in real-time with production-ready MLOps practices.

## 🎯 Project Overview

This system demonstrates advanced ML engineering skills including:
- **Ensemble ML Model**: Random Forest + XGBoost achieving >90% accuracy
- **Real-time API**: FastAPI service processing transactions in <500ms
- **MLOps Pipeline**: MLflow for experiment tracking and model versioning
- **Monitoring Dashboard**: Streamlit interface for real-time insights
- **Production Deployment**: Docker containerization and CI/CD ready

## 🏗️ Architecture

```
fraud_detection/
├── data/                   # Data storage and generation
├── models/                 # Trained model artifacts
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # ML model training and inference
│   ├── api/               # FastAPI application
│   ├── dashboard/         # Streamlit dashboard
│   └── mlops/             # MLOps utilities
├── tests/                 # Unit and integration tests
├── docker/                # Docker configuration
├── mlruns/                # MLflow experiment tracking
└── docs/                  # Documentation
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <your-repo-url>
cd fraud_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python src/data/generate_data.py
```

### 3. Train Model
```bash
python src/models/train_model.py
```

### 4. Start Services
```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit dashboard (in new terminal)
streamlit run src/dashboard/app.py --server.port 8501
```

### 5. Docker Deployment
```bash
# Build and run with Docker
docker-compose up --build
```

## 📊 Model Performance

- **Accuracy**: >90%
- **Precision**: >85%
- **Recall**: >88%
- **F1-Score**: >86%
- **Response Time**: <500ms

## 🔧 API Endpoints

### Predict Fraud
```bash
POST /predict
{
  "amount": 100.50,
  "merchant_category": "online_retail",
  "hour": 14,
  "day_of_week": 2,
  "distance_from_home": 25.3,
  "distance_from_last_transaction": 0.5,
  "ratio_to_median_purchase_price": 1.2,
  "repeat_retailer": false,
  "used_chip": true,
  "used_pin_number": false,
  "online_order": true
}
```

### Health Check
```bash
GET /health
```

### Model Metrics
```bash
GET /metrics
```

## 📈 Dashboard Features

- Real-time transaction monitoring
- Model performance metrics
- Fraud detection statistics
- Interactive visualizations
- Historical trend analysis

## 🛠️ MLOps Features

- **Experiment Tracking**: MLflow integration
- **Model Versioning**: Automatic model registry
- **Performance Monitoring**: Real-time metrics
- **A/B Testing**: Model comparison capabilities
- **Reproducibility**: Environment and dependency tracking

## 🐳 Docker Deployment

The system is containerized for easy deployment:

```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.prod.yml up
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📝 Documentation

- [API Documentation](docs/api.md)
- [Model Architecture](docs/model.md)
- [Deployment Guide](docs/deployment.md)
- [MLOps Pipeline](docs/mlops.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🎥 Demo Video

[Link to demo video showcasing the system in action]

---

**Built with ❤️ for demonstrating advanced ML engineering skills**
