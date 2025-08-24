"""
Credit Card Fraud Detection - FastAPI Application

Real-time API for fraud detection with sub-500ms response times.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import time
import os
import sys
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.fraud_detector import FraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection API with ensemble ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
fraud_detector = None
model_loaded = False
prediction_history = []
performance_metrics = {
    "total_predictions": 0,
    "fraud_predictions": 0,
    "avg_response_time": 0.0,
    "last_updated": None
}

# Pydantic models for request/response
class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., description="Merchant category")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0-6)")
    distance_from_home: float = Field(..., ge=0, description="Distance from home in miles")
    distance_from_last_transaction: float = Field(..., ge=0, description="Distance from last transaction")
    ratio_to_median_purchase_price: float = Field(..., description="Ratio to median purchase price")
    repeat_retailer: bool = Field(..., description="Is repeat retailer")
    used_chip: bool = Field(..., description="Used chip for transaction")
    used_pin_number: bool = Field(..., description="Used PIN number")
    online_order: bool = Field(..., description="Is online order")

class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    confidence: str
    response_time_ms: float
    timestamp: str
    risk_factors: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict
    uptime: str
    version: str

class MetricsResponse(BaseModel):
    total_predictions: int
    fraud_predictions: int
    fraud_rate: float
    avg_response_time: float
    recent_predictions: List[Dict]
    performance_trend: Dict

def load_model():
    """Load the trained fraud detection model."""
    global fraud_detector, model_loaded
    
    try:
        model_path = "models/fraud_detector.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        fraud_detector = FraudDetector()
        fraud_detector.load_model(model_path)
        model_loaded = True
        logger.info("✅ Fraud detection model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_transaction(transaction: TransactionRequest) -> np.ndarray:
    """Preprocess transaction data for prediction."""
    
    # Convert to DataFrame
    df = pd.DataFrame([transaction.dict()])
    
    # Preprocess using the same pipeline as training
    df_processed = fraud_detector.preprocess_features(df)
    
    # Convert to numpy array
    X = df_processed.values
    
    return X

def get_risk_factors(transaction: TransactionRequest, fraud_prob: float) -> List[str]:
    """Identify risk factors for the transaction."""
    
    risk_factors = []
    
    # Amount-based risks
    if transaction.amount > 1000:
        risk_factors.append("High transaction amount")
    
    # Time-based risks
    if transaction.hour < 6 or transaction.hour > 22:
        risk_factors.append("Unusual transaction time")
    
    # Location-based risks
    if transaction.distance_from_home > 50:
        risk_factors.append("Far from home location")
    
    # Merchant-based risks
    high_risk_categories = ['jewelry', 'electronics', 'online_retail']
    if transaction.merchant_category in high_risk_categories:
        risk_factors.append(f"High-risk merchant category: {transaction.merchant_category}")
    
    # Online transaction risks
    if transaction.online_order:
        risk_factors.append("Online transaction")
    
    # Price ratio risks
    if transaction.ratio_to_median_purchase_price > 3:
        risk_factors.append("Unusually high purchase ratio")
    
    return risk_factors

def get_confidence_level(fraud_prob: float) -> str:
    """Get confidence level based on fraud probability."""
    
    if fraud_prob < 0.2:
        return "LOW"
    elif fraud_prob < 0.5:
        return "MEDIUM"
    elif fraud_prob < 0.8:
        return "HIGH"
    else:
        return "VERY_HIGH"

def update_performance_metrics(response_time: float, is_fraud: bool):
    """Update performance metrics."""
    global performance_metrics, prediction_history
    
    # Update counters
    performance_metrics["total_predictions"] += 1
    if is_fraud:
        performance_metrics["fraud_predictions"] += 1
    
    # Update average response time
    current_avg = performance_metrics["avg_response_time"]
    total_preds = performance_metrics["total_predictions"]
    performance_metrics["avg_response_time"] = (
        (current_avg * (total_preds - 1) + response_time) / total_preds
    )
    
    performance_metrics["last_updated"] = datetime.now().isoformat()
    
    # Keep only recent predictions in history
    if len(prediction_history) > 100:
        prediction_history.pop(0)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("🚀 Starting Fraud Detection API...")
    
    # Load model
    if load_model():
        logger.info("✅ API startup completed successfully")
    else:
        logger.error("❌ Failed to load model during startup")

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    # Calculate uptime (simplified)
    uptime = "Running"
    
    model_info = {}
    if model_loaded and fraud_detector:
        model_info = fraud_detector.get_model_info()
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_info=model_info,
        uptime=uptime,
        version="1.0.0"
    )

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionRequest):
    """Predict fraud for a credit card transaction."""
    
    if not model_loaded or fraud_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess transaction
        X = preprocess_transaction(transaction)
        
        # Make prediction
        fraud_probability = fraud_detector.predict_proba(X)[0]
        is_fraud = fraud_probability > 0.5
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Generate transaction ID
        transaction_id = f"txn_{int(time.time() * 1000)}"
        
        # Get risk factors and confidence
        risk_factors = get_risk_factors(transaction, fraud_probability)
        confidence = get_confidence_level(fraud_probability)
        
        # Create prediction response
        prediction = FraudPrediction(
            transaction_id=transaction_id,
            fraud_probability=float(fraud_probability),
            is_fraud=bool(is_fraud),
            confidence=confidence,
            response_time_ms=round(response_time, 2),
            timestamp=datetime.now().isoformat(),
            risk_factors=risk_factors
        )
        
        # Update performance metrics
        update_performance_metrics(response_time, is_fraud)
        
        # Add to prediction history
        prediction_history.append({
            "transaction_id": transaction_id,
            "amount": transaction.amount,
            "fraud_probability": float(fraud_probability),
            "is_fraud": bool(is_fraud),
            "response_time_ms": round(response_time, 2),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Prediction completed in {response_time:.2f}ms - Fraud: {is_fraud}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get API performance metrics."""
    
    fraud_rate = 0.0
    if performance_metrics["total_predictions"] > 0:
        fraud_rate = performance_metrics["fraud_predictions"] / performance_metrics["total_predictions"]
    
    # Calculate performance trend (simplified)
    performance_trend = {
        "avg_response_time_trend": "stable",
        "fraud_rate_trend": "stable"
    }
    
    return MetricsResponse(
        total_predictions=performance_metrics["total_predictions"],
        fraud_predictions=performance_metrics["fraud_predictions"],
        fraud_rate=round(fraud_rate, 4),
        avg_response_time=round(performance_metrics["avg_response_time"], 2),
        recent_predictions=prediction_history[-10:],  # Last 10 predictions
        performance_trend=performance_trend
    )

@app.post("/predict/batch")
async def predict_fraud_batch(transactions: List[TransactionRequest]):
    """Predict fraud for multiple transactions (batch processing)."""
    
    if not model_loaded or fraud_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    results = []
    
    try:
        for i, transaction in enumerate(transactions):
            # Preprocess transaction
            X = preprocess_transaction(transaction)
            
            # Make prediction
            fraud_probability = fraud_detector.predict_proba(X)[0]
            is_fraud = fraud_probability > 0.5
            
            # Get risk factors and confidence
            risk_factors = get_risk_factors(transaction, fraud_probability)
            confidence = get_confidence_level(fraud_probability)
            
            result = {
                "transaction_index": i,
                "fraud_probability": float(fraud_probability),
                "is_fraud": bool(is_fraud),
                "confidence": confidence,
                "risk_factors": risk_factors
            }
            
            results.append(result)
            
            # Update performance metrics
            response_time = (time.time() - start_time) * 1000
            update_performance_metrics(response_time, is_fraud)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "predictions": results,
            "total_transactions": len(transactions),
            "total_time_ms": round(total_time, 2),
            "avg_time_per_transaction_ms": round(total_time / len(transactions), 2)
        }
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    
    if not model_loaded or fraud_detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    model_info = fraud_detector.get_model_info()
    
    return {
        "model_info": model_info,
        "feature_count": len(model_info.get("feature_names", [])),
        "model_weights": model_info.get("model_weights", {}),
        "top_features": list(model_info.get("feature_names", []))[:10]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
