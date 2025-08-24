# Fraud Detection System - Deployment Guide

This guide covers deploying the fraud detection system in various environments, from local development to production.

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)

## Local Development Setup

### Prerequisites

- Python 3.9+
- pip
- Git

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd fraud_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Data and Train Model

```bash
# Generate training data
python src/data/generate_data.py

# Train the model
python src/models/train_model.py
```

### Step 3: Start Services

```bash
# Terminal 1: Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit dashboard
streamlit run src/dashboard/app.py --server.port 8501

# Terminal 3: Start MLflow UI (optional)
mlflow ui --port 5000
```

### Step 4: Verify Installation

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs
- MLflow UI: http://localhost:5000

## Docker Deployment

### Quick Start with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Docker Containers

```bash
# Build the image
docker build -t fraud-detection .

# Run API container
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fraud-detection

# Run dashboard container
docker run -d \
  --name fraud-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  fraud-detection \
  streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Production Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  fraud-api:
    build: .
    container_name: fraud-detection-api-prod
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./mlruns:/app/mlruns
    environment:
      - PYTHONPATH=/app
      - ENVIRONMENT=production
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  fraud-dashboard:
    build: .
    container_name: fraud-detection-dashboard-prod
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
      - ENVIRONMENT=production
    command: streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
    depends_on:
      - fraud-api
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: fraud-detection-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - fraud-api
      - fraud-dashboard
    restart: unless-stopped

networks:
  default:
    name: fraud-detection-network
```

## Production Deployment

### Environment Variables

Create `.env.production`:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# Model Configuration
MODEL_PATH=models/fraud_detector.pkl
DATA_PATH=data/

# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=fraud_detection

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Database (if using)
DATABASE_URL=postgresql://user:password@localhost/fraud_db

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

### Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server fraud-api:8000;
    }

    upstream dashboard_backend {
        server fraud-dashboard:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;

        # API endpoints
        location /api/ {
            proxy_pass http://api_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Dashboard
        location / {
            proxy_pass http://dashboard_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Systemd Service Files

Create `/etc/systemd/system/fraud-detection-api.service`:

```ini
[Unit]
Description=Fraud Detection API
After=network.target

[Service]
Type=simple
User=fraud-detection
WorkingDirectory=/opt/fraud-detection
Environment=PATH=/opt/fraud-detection/venv/bin
ExecStart=/opt/fraud-detection/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Deployment Script

Create `deploy.sh`:

```bash
#!/bin/bash

# Production deployment script
set -e

echo "🚀 Starting production deployment..."

# Update code
git pull origin main

# Install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Generate data and train model (if needed)
python src/data/generate_data.py
python src/models/train_model.py

# Restart services
sudo systemctl restart fraud-detection-api
sudo systemctl restart fraud-detection-dashboard

# Check health
sleep 10
curl -f http://localhost:8000/health || exit 1

echo "✅ Deployment completed successfully!"
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

1. **Create ECR Repository**:
```bash
aws ecr create-repository --repository-name fraud-detection
```

2. **Build and Push Image**:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t fraud-detection .
docker tag fraud-detection:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
```

3. **Create ECS Task Definition**:
```json
{
  "family": "fraud-detection",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "fraud-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "command": ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fraud-detection",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Using AWS Lambda

Create `lambda_function.py`:

```python
import json
import os
import sys
from src.models.fraud_detector import FraudDetector

# Load model
detector = FraudDetector()
detector.load_model('/opt/models/fraud_detector.pkl')

def lambda_handler(event, context):
    try:
        # Parse request
        transaction = json.loads(event['body'])
        
        # Make prediction
        import pandas as pd
        df = pd.DataFrame([transaction])
        df_processed = detector.preprocess_features(df)
        X = df_processed.values
        
        fraud_prob = detector.predict_proba(X)[0]
        is_fraud = fraud_prob > 0.5
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(is_fraud),
                'confidence': 'HIGH' if fraud_prob > 0.8 else 'MEDIUM' if fraud_prob > 0.5 else 'LOW'
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Platform

#### Using Cloud Run

1. **Build and Deploy**:
```bash
# Build image
gcloud builds submit --tag gcr.io/PROJECT_ID/fraud-detection

# Deploy to Cloud Run
gcloud run deploy fraud-detection-api \
  --image gcr.io/PROJECT_ID/fraud-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

#### Using Cloud Functions

Create `main.py`:

```python
import functions_framework
from src.models.fraud_detector import FraudDetector
import json

detector = FraudDetector()
detector.load_model('models/fraud_detector.pkl')

@functions_framework.http
def predict_fraud(request):
    try:
        request_json = request.get_json()
        transaction = request_json['transaction']
        
        import pandas as pd
        df = pd.DataFrame([transaction])
        df_processed = detector.preprocess_features(df)
        X = df_processed.values
        
        fraud_prob = detector.predict_proba(X)[0]
        is_fraud = fraud_prob > 0.5
        
        return json.dumps({
            'fraud_probability': float(fraud_prob),
            'is_fraud': bool(is_fraud)
        })
    except Exception as e:
        return json.dumps({'error': str(e)}), 500
```

### Azure

#### Using Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image fraud-detection .

# Deploy to Container Instances
az container create \
  --resource-group myResourceGroup \
  --name fraud-detection \
  --image myregistry.azurecr.io/fraud-detection:latest \
  --ports 8000 \
  --dns-name-label fraud-detection \
  --registry-login-server myregistry.azurecr.io \
  --registry-username <username> \
  --registry-password <password>
```

## Monitoring and Maintenance

### Health Checks

```bash
# API health check
curl -f http://localhost:8000/health

# Metrics endpoint
curl http://localhost:8000/metrics

# Model info
curl http://localhost:8000/model/info
```

### Logging

Configure logging in `src/api/main.py`:

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/api.log', maxBytes=10000000, backupCount=5),
        logging.StreamHandler()
    ]
)
```

### Monitoring with Prometheus

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
PREDICTION_COUNTER = Counter('fraud_predictions_total', 'Total fraud predictions')
FRAUD_COUNTER = Counter('fraud_detected_total', 'Total fraud detected')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Backup models
tar -czf backups/models_$(date +%Y%m%d_%H%M%S).tar.gz models/

# Backup data
tar -czf backups/data_$(date +%Y%m%d_%H%M%S).tar.gz data/

# Backup MLflow experiments
tar -czf backups/mlruns_$(date +%Y%m%d_%H%M%S).tar.gz mlruns/

# Keep only last 7 days of backups
find backups/ -name "*.tar.gz" -mtime +7 -delete
```

## Troubleshooting

### Common Issues

1. **Model not loading**:
   ```bash
   # Check if model file exists
   ls -la models/fraud_detector.pkl
   
   # Regenerate model
   python src/models/train_model.py
   ```

2. **API not responding**:
   ```bash
   # Check if API is running
   ps aux | grep uvicorn
   
   # Check logs
   tail -f logs/api.log
   
   # Restart API
   sudo systemctl restart fraud-detection-api
   ```

3. **Memory issues**:
   ```bash
   # Monitor memory usage
   htop
   
   # Increase swap space
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Docker issues**:
   ```bash
   # Clean up Docker
   docker system prune -a
   
   # Rebuild containers
   docker-compose down
   docker-compose up --build
   ```

### Performance Optimization

1. **Model optimization**:
   - Use model quantization
   - Implement caching
   - Use async processing

2. **API optimization**:
   - Enable gzip compression
   - Use connection pooling
   - Implement rate limiting

3. **Database optimization** (if using):
   - Add indexes
   - Use connection pooling
   - Implement caching

### Security Considerations

1. **API Security**:
   - Implement authentication
   - Use HTTPS
   - Add rate limiting
   - Validate input data

2. **Model Security**:
   - Secure model storage
   - Implement model versioning
   - Monitor for adversarial attacks

3. **Infrastructure Security**:
   - Use VPCs
   - Implement firewall rules
   - Regular security updates
   - Monitor access logs

## Support

For deployment issues:

1. Check the logs: `docker-compose logs -f`
2. Verify configuration files
3. Test individual components
4. Check system resources
5. Review security settings

For additional help, refer to the project documentation or create an issue in the GitHub repository.
