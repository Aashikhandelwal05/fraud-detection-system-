#!/usr/bin/env python3
"""
Fraud Detection System Demo

This script demonstrates the complete fraud detection system by:
1. Generating sample data
2. Training the model
3. Making predictions
4. Showing performance metrics
"""

import os
import sys
import time
import requests
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n📋 Step {step}: {description}")
    print("-" * 40)

def check_dependencies():
    """Check if required packages are installed."""
    print_step(1, "Checking Dependencies")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import fastapi
        import streamlit
        import mlflow
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def generate_data():
    """Generate sample transaction data."""
    print_step(2, "Generating Sample Data")
    
    try:
        from data.generate_data import FraudDataGenerator
        
        generator = FraudDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=10000)  # Smaller dataset for demo
        
        print(f"✅ Generated {len(df):,} transaction records")
        print(f"📊 Fraud rate: {labels.mean():.2%}")
        print(f"📁 Data saved to: data/credit_card_transactions.csv")
        
        return True
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False

def train_model():
    """Train the fraud detection model."""
    print_step(3, "Training Fraud Detection Model")
    
    try:
        from models.train_model import main as train_main
        train_main()
        print("✅ Model training completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def test_api():
    """Test the API with sample predictions."""
    print_step(4, "Testing API Predictions")
    
    # Sample transactions for testing
    test_transactions = [
        {
            "amount": 25.0,
            "merchant_category": "grocery_store",
            "hour": 14,
            "day_of_week": 2,
            "distance_from_home": 2.0,
            "distance_from_last_transaction": 1.0,
            "ratio_to_median_purchase_price": 0.8,
            "repeat_retailer": True,
            "used_chip": True,
            "used_pin_number": True,
            "online_order": False
        },
        {
            "amount": 1500.0,
            "merchant_category": "jewelry",
            "hour": 2,
            "day_of_week": 6,
            "distance_from_home": 100.0,
            "distance_from_last_transaction": 50.0,
            "ratio_to_median_purchase_price": 5.0,
            "repeat_retailer": False,
            "used_chip": False,
            "used_pin_number": False,
            "online_order": True
        },
        {
            "amount": 500.0,
            "merchant_category": "electronics",
            "hour": 20,
            "day_of_week": 5,
            "distance_from_home": 30.0,
            "distance_from_last_transaction": 25.0,
            "ratio_to_median_purchase_price": 2.5,
            "repeat_retailer": False,
            "used_chip": True,
            "used_pin_number": False,
            "online_order": True
        }
    ]
    
    api_url = "http://localhost:8000"
    
    # Test API health
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy")
        else:
            print("⚠️ API is running but health check failed")
    except requests.exceptions.RequestException:
        print("❌ API is not running. Please start the API first:")
        print("   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
        return False
    
    # Test predictions
    print("\n🔮 Testing Fraud Predictions:")
    print("-" * 50)
    
    for i, transaction in enumerate(test_transactions, 1):
        try:
            start_time = time.time()
            response = requests.post(f"{api_url}/predict", json=transaction, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                
                # Determine transaction type
                if transaction['amount'] < 100:
                    transaction_type = "Small Purchase"
                elif transaction['amount'] < 1000:
                    transaction_type = "Medium Purchase"
                else:
                    transaction_type = "Large Purchase"
                
                print(f"\n📊 Transaction {i} ({transaction_type}):")
                print(f"   Amount: ${transaction['amount']:.2f}")
                print(f"   Merchant: {transaction['merchant_category']}")
                print(f"   Time: {transaction['hour']}:00 on day {transaction['day_of_week']}")
                print(f"   Fraud Probability: {result['fraud_probability']:.2%}")
                print(f"   Prediction: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Response Time: {result['response_time_ms']:.1f}ms")
                
                if result['risk_factors']:
                    print(f"   Risk Factors: {', '.join(result['risk_factors'])}")
                
            else:
                print(f"❌ Prediction {i} failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing prediction {i}: {e}")
    
    return True

def show_metrics():
    """Show API performance metrics."""
    print_step(5, "API Performance Metrics")
    
    api_url = "http://localhost:8000"
    
    try:
        # Get metrics
        response = requests.get(f"{api_url}/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.json()
            
            print("📈 Performance Overview:")
            print(f"   Total Predictions: {metrics['total_predictions']:,}")
            print(f"   Fraud Detected: {metrics['fraud_predictions']:,}")
            print(f"   Fraud Rate: {metrics['fraud_rate']:.2%}")
            print(f"   Average Response Time: {metrics['avg_response_time']:.1f}ms")
            
            # Get model info
            response = requests.get(f"{api_url}/model/info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                
                print("\n🤖 Model Information:")
                print(f"   Status: {model_info['model_info']['status']}")
                print(f"   Features: {model_info['feature_count']}")
                print(f"   Model Weights - RF: {model_info['model_weights']['rf']:.3f}, XGB: {model_info['model_weights']['xgb']:.3f}")
                
                print("\n🔝 Top Features:")
                for i, feature in enumerate(model_info['top_features'][:5], 1):
                    print(f"   {i}. {feature}")
            
        else:
            print("❌ Failed to get metrics")
            
    except Exception as e:
        print(f"❌ Error getting metrics: {e}")

def show_next_steps():
    """Show next steps for the user."""
    print_step(6, "Next Steps")
    
    print("🎉 Demo completed successfully!")
    print("\n📋 What you can do next:")
    print("   1. 🖥️  Start the Streamlit Dashboard:")
    print("      streamlit run src/dashboard/app.py --server.port 8501")
    print("   2. 🔧 Explore the API documentation:")
    print("      http://localhost:8000/docs")
    print("   3. 📊 View MLflow experiments:")
    print("      mlflow ui --port 5000")
    print("   4. 🐳 Deploy with Docker:")
    print("      docker-compose up --build")
    print("   5. 🧪 Run tests:")
    print("      pytest tests/")
    
    print("\n🔗 Quick Links:")
    print("   • API: http://localhost:8000")
    print("   • Dashboard: http://localhost:8501")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • MLflow UI: http://localhost:5000")

def main():
    """Main demo function."""
    print_header("Fraud Detection System Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the right directory
    if not os.path.exists('src'):
        print("❌ Please run this script from the project root directory")
        return
    
    # Run demo steps
    steps = [
        ("Dependencies", check_dependencies),
        ("Data Generation", generate_data),
        ("Model Training", train_model),
        ("API Testing", test_api),
        ("Metrics", show_metrics),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Demo failed at step: {step_name}")
            return
    
    show_next_steps()
    
    print(f"\n✅ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()


