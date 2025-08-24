"""
Credit Card Fraud Detection - Model Training Script

This script trains the ensemble fraud detection model and tracks
experiments using MLflow for MLOps best practices.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.generate_data import FraudDataGenerator
from models.fraud_detector import FraudDetector

def setup_mlflow():
    """Setup MLflow tracking."""
    
    # Set MLflow tracking URI (local file system)
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment name
    experiment_name = "fraud_detection"
    mlflow.set_experiment(experiment_name)
    
    print(f"🔧 MLflow tracking setup complete")
    print(f"📁 Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"🧪 Experiment: {experiment_name}")

def log_model_metrics(metrics: dict, feature_importance: dict, model_weights: dict):
    """Log model metrics to MLflow."""
    
    # Log performance metrics
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
        print(f"📊 {metric_name}: {metric_value:.4f}")
    
    # Log model weights
    for model_name, weight in model_weights.items():
        mlflow.log_metric(f"weight_{model_name}", weight)
    
    # Log feature importance (top 10)
    top_features = dict(list(feature_importance.items())[:10])
    for feature, importance in top_features.items():
        mlflow.log_metric(f"importance_{feature}", importance)

def train_and_log_model():
    """Train model and log everything to MLflow."""
    
    with mlflow.start_run(run_name=f"fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print("🚀 Starting Fraud Detection Model Training with MLflow...")
        
        # Log parameters
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("use_smote", True)
        mlflow.log_param("n_samples", 100000)
        
        # Generate data
        print("📊 Generating training data...")
        generator = FraudDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=100000)
        
        # Save data artifact
        data_path = "data/credit_card_transactions.csv"
        df.to_csv(data_path, index=False)
        mlflow.log_artifact(data_path, "training_data")
        
        # Initialize and train model
        print("🤖 Training ensemble fraud detection model...")
        detector = FraudDetector(random_state=42)
        
        # Train model
        results = detector.train(df, target_col='fraud', test_size=0.2, use_smote=True)
        
        # Log metrics
        print("📈 Logging model metrics...")
        log_model_metrics(
            results['metrics'], 
            results['feature_importance'], 
            results['model_weights']
        )
        
        # Log model artifacts
        print("💾 Saving model artifacts...")
        model_path = "models/fraud_detector.pkl"
        detector.save_model(model_path)
        mlflow.log_artifact(model_path, "model")
        
        # Log feature importance as artifact
        importance_df = pd.DataFrame(
            list(results['feature_importance'].items()),
            columns=['feature', 'importance']
        )
        importance_path = "models/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, "feature_importance")
        
        # Log model with MLflow
        mlflow.sklearn.log_model(detector.rf_model, "random_forest")
        mlflow.xgboost.log_model(detector.xgb_model, "xgboost")
        
        # Log model info
        model_info = detector.get_model_info()
        mlflow.log_dict(model_info, "model_info.json")
        
        print("✅ Model training and logging completed!")
        
        return detector, results

def main():
    """Main training function."""
    
    try:
        # Setup MLflow
        setup_mlflow()
        
        # Train and log model
        detector, results = train_and_log_model()
        
        # Print final results
        print("\n🎉 Training Summary:")
        print("=" * 50)
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Precision: {results['metrics']['precision']:.4f}")
        print(f"Recall: {results['metrics']['recall']:.4f}")
        print(f"F1-Score: {results['metrics']['f1_score']:.4f}")
        print(f"ROC-AUC: {results['metrics']['roc_auc']:.4f}")
        print(f"Average Precision: {results['metrics']['average_precision']:.4f}")
        
        print(f"\n📊 Model Weights:")
        for model, weight in results['model_weights'].items():
            print(f"  {model.upper()}: {weight:.3f}")
        
        print(f"\n🔝 Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        print(f"\n📁 Model saved to: models/fraud_detector.pkl")
        print(f"📊 MLflow run completed. Check mlruns/ for experiment tracking.")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
