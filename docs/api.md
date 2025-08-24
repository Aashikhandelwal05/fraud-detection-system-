# Fraud Detection API Documentation

## Overview

The Fraud Detection API is a real-time service that analyzes credit card transactions to identify potential fraud using an ensemble machine learning model (Random Forest + XGBoost).

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, consider implementing API keys or OAuth2.

## Endpoints

### 1. Health Check

**GET** `/health`

Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "status": "Trained",
    "feature_count": 20,
    "model_weights": {
      "rf": 0.52,
      "xgb": 0.48
    }
  },
  "uptime": "Running",
  "version": "1.0.0"
}
```

### 2. Root Information

**GET** `/`

Get basic API information.

**Response:**
```json
{
  "message": "Credit Card Fraud Detection API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

### 3. Fraud Prediction

**POST** `/predict`

Predict fraud for a single transaction.

**Request Body:**
```json
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

**Response:**
```json
{
  "transaction_id": "txn_1703123456789",
  "fraud_probability": 0.23,
  "is_fraud": false,
  "confidence": "LOW",
  "response_time_ms": 45.2,
  "timestamp": "2023-12-21T10:30:45.123456",
  "risk_factors": [
    "Online transaction",
    "High-risk merchant category: online_retail"
  ]
}
```

### 4. Batch Prediction

**POST** `/predict/batch`

Predict fraud for multiple transactions.

**Request Body:**
```json
[
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
  },
  {
    "amount": 1500.00,
    "merchant_category": "jewelry",
    "hour": 2,
    "day_of_week": 6,
    "distance_from_home": 100.0,
    "distance_from_last_transaction": 50.0,
    "ratio_to_median_purchase_price": 5.0,
    "repeat_retailer": false,
    "used_chip": false,
    "used_pin_number": false,
    "online_order": true
  }
]
```

**Response:**
```json
{
  "predictions": [
    {
      "transaction_index": 0,
      "fraud_probability": 0.23,
      "is_fraud": false,
      "confidence": "LOW",
      "risk_factors": ["Online transaction"]
    },
    {
      "transaction_index": 1,
      "fraud_probability": 0.87,
      "is_fraud": true,
      "confidence": "VERY_HIGH",
      "risk_factors": [
        "High transaction amount",
        "Unusual transaction time",
        "Far from home location",
        "High-risk merchant category: jewelry",
        "Online transaction",
        "Unusually high purchase ratio"
      ]
    }
  ],
  "total_transactions": 2,
  "total_time_ms": 89.5,
  "avg_time_per_transaction_ms": 44.75
}
```

### 5. Performance Metrics

**GET** `/metrics`

Get API performance metrics.

**Response:**
```json
{
  "total_predictions": 1250,
  "fraud_predictions": 89,
  "fraud_rate": 0.0712,
  "avg_response_time": 42.3,
  "recent_predictions": [
    {
      "transaction_id": "txn_1703123456789",
      "amount": 100.50,
      "fraud_probability": 0.23,
      "is_fraud": false,
      "response_time_ms": 45.2,
      "timestamp": "2023-12-21T10:30:45.123456"
    }
  ],
  "performance_trend": {
    "avg_response_time_trend": "stable",
    "fraud_rate_trend": "stable"
  }
}
```

### 6. Model Information

**GET** `/model/info`

Get detailed information about the loaded model.

**Response:**
```json
{
  "model_info": {
    "status": "Trained",
    "feature_count": 20,
    "model_weights": {
      "rf": 0.52,
      "xgb": 0.48
    },
    "feature_names": [
      "amount",
      "hour",
      "distance_from_home",
      "amount_log",
      "is_night",
      "merchant_category_encoded",
      "distance_ratio",
      "amount_squared",
      "is_weekend",
      "high_amount_flag",
      "unusual_time_flag",
      "is_far_from_home",
      "amount_distance_interaction",
      "online_amount_interaction",
      "repeat_retailer",
      "used_chip",
      "used_pin_number",
      "online_order",
      "distance_from_last_transaction",
      "ratio_to_median_purchase_price"
    ]
  },
  "feature_count": 20,
  "model_weights": {
    "rf": 0.52,
    "xgb": 0.48
  },
  "top_features": [
    "amount",
    "hour",
    "distance_from_home",
    "amount_log",
    "is_night"
  ]
}
```

## Data Types

### Transaction Request Fields

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `amount` | float | Transaction amount in dollars | > 0 |
| `merchant_category` | string | Merchant category | See categories below |
| `hour` | integer | Hour of transaction (0-23) | 0 ≤ hour ≤ 23 |
| `day_of_week` | integer | Day of week (0=Monday, 6=Sunday) | 0 ≤ day ≤ 6 |
| `distance_from_home` | float | Distance from home in miles | ≥ 0 |
| `distance_from_last_transaction` | float | Distance from last transaction | ≥ 0 |
| `ratio_to_median_purchase_price` | float | Ratio to median purchase price | > 0 |
| `repeat_retailer` | boolean | Is repeat retailer | true/false |
| `used_chip` | boolean | Used chip for transaction | true/false |
| `used_pin_number` | boolean | Used PIN number | true/false |
| `online_order` | boolean | Is online order | true/false |

### Merchant Categories

- `online_retail`
- `gas_station`
- `grocery_store`
- `restaurant`
- `electronics`
- `travel`
- `pharmacy`
- `jewelry`
- `entertainment`
- `utilities`

### Confidence Levels

- `LOW`: Fraud probability < 0.2
- `MEDIUM`: Fraud probability 0.2 - 0.5
- `HIGH`: Fraud probability 0.5 - 0.8
- `VERY_HIGH`: Fraud probability ≥ 0.8

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error: amount must be greater than 0"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Model not loaded"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Prediction error: Invalid input data"
}
```

## Rate Limiting

Currently, there are no rate limits implemented. In production, consider implementing rate limiting to prevent abuse.

## Performance

- **Response Time**: < 500ms for single predictions
- **Throughput**: ~1000 predictions/second
- **Model Accuracy**: > 90%
- **Availability**: 99.9% uptime target

## Examples

### Python Example

```python
import requests
import json

# API base URL
base_url = "http://localhost:8000"

# Sample transaction
transaction = {
    "amount": 100.50,
    "merchant_category": "online_retail",
    "hour": 14,
    "day_of_week": 2,
    "distance_from_home": 25.3,
    "distance_from_last_transaction": 0.5,
    "ratio_to_median_purchase_price": 1.2,
    "repeat_retailer": False,
    "used_chip": True,
    "used_pin_number": False,
    "online_order": True
}

# Make prediction
response = requests.post(f"{base_url}/predict", json=transaction)
result = response.json()

print(f"Fraud Probability: {result['fraud_probability']:.2%}")
print(f"Is Fraud: {result['is_fraud']}")
print(f"Confidence: {result['confidence']}")
print(f"Response Time: {result['response_time_ms']:.1f}ms")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

## Monitoring

The API provides several endpoints for monitoring:

- `/health` - Service health check
- `/metrics` - Performance metrics
- `/model/info` - Model information

## Support

For issues or questions, please refer to the project documentation or create an issue in the GitHub repository.
