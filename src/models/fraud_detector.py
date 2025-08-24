"""
Credit Card Fraud Detection - ML Model Module

This module contains the ensemble fraud detection model combining
Random Forest and XGBoost for high-accuracy fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import os
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class FraudDetector:
    """Ensemble fraud detection model combining Random Forest and XGBoost."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the fraud detector."""
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Initialize models
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        self.is_trained = False
        self.feature_names = None
        self.model_weights = {'rf': 0.5, 'xgb': 0.5}  # Equal weights initially
        
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for model training."""
        
        df_processed = df.copy()
        
        # Encode categorical variables
        if 'merchant_category' in df_processed.columns:
            df_processed['merchant_category_encoded'] = self.label_encoder.fit_transform(
                df_processed['merchant_category']
            )
            df_processed = df_processed.drop('merchant_category', axis=1)
        
        # Convert boolean columns to int
        boolean_columns = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
        for col in boolean_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        return df_processed
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for better fraud detection."""
        
        # Time-based features
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Amount-based features
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_squared'] = df['amount'] ** 2
        
        # Distance-based features
        df['distance_ratio'] = df['distance_from_last_transaction'] / (df['distance_from_home'] + 1e-6)
        df['is_far_from_home'] = (df['distance_from_home'] > 50).astype(int)
        
        # Risk indicators
        df['high_amount_flag'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['unusual_time_flag'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # Interaction features
        df['amount_distance_interaction'] = df['amount'] * df['distance_from_home']
        df['online_amount_interaction'] = df['online_order'] * df['amount']
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'fraud') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training."""
        
        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_col: str = 'fraud', 
              test_size: float = 0.2, use_smote: bool = True) -> Dict[str, Any]:
        """Train the ensemble fraud detection model."""
        
        print("🔄 Starting model training...")
        
        # Preprocess features
        df_processed = self.preprocess_features(df)
        
        # Prepare data
        X, y = self.prepare_data(df_processed, target_col)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        if use_smote:
            print("📊 Applying SMOTE for class balance...")
            smote = SMOTE(random_state=self.random_state)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("🌲 Training Random Forest...")
        self.rf_model.fit(X_train_scaled, y_train_balanced)
        
        # Train XGBoost
        print("🚀 Training XGBoost...")
        self.xgb_model.fit(X_train_scaled, y_train_balanced)
        
        # Evaluate individual models
        rf_pred = self.rf_model.predict(X_test_scaled)
        xgb_pred = self.xgb_model.predict(X_test_scaled)
        
        # Calculate model weights based on performance
        rf_score = roc_auc_score(y_test, self.rf_model.predict_proba(X_test_scaled)[:, 1])
        xgb_score = roc_auc_score(y_test, self.xgb_model.predict_proba(X_test_scaled)[:, 1])
        
        total_score = rf_score + xgb_score
        self.model_weights = {
            'rf': rf_score / total_score,
            'xgb': xgb_score / total_score
        }
        
        print(f"📈 Model Weights - RF: {self.model_weights['rf']:.3f}, XGB: {self.model_weights['xgb']:.3f}")
        
        # Ensemble prediction
        ensemble_pred = self._ensemble_predict(X_test_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, ensemble_pred, X_test_scaled)
        
        self.is_trained = True
        
        print("✅ Model training completed!")
        
        return {
            'metrics': metrics,
            'feature_importance': self._get_feature_importance(),
            'model_weights': self.model_weights
        }
    
    def _ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction using weighted average."""
        
        rf_proba = self.rf_model.predict_proba(X)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        ensemble_proba = (
            self.model_weights['rf'] * rf_proba + 
            self.model_weights['xgb'] * xgb_proba
        )
        
        return (ensemble_proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud probability using ensemble."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_scaled)[:, 1]
        
        ensemble_proba = (
            self.model_weights['rf'] * rf_proba + 
            self.model_weights['xgb'] * xgb_proba
        )
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict fraud labels using ensemble."""
        
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          X_test: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, self.predict_proba(X_test)),
            'average_precision': average_precision_score(y_true, self.predict_proba(X_test))
        }
        
        return metrics
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from both models."""
        
        rf_importance = self.rf_model.feature_importances_
        xgb_importance = self.xgb_model.feature_importances_
        
        # Weighted average importance
        weighted_importance = (
            self.model_weights['rf'] * rf_importance + 
            self.model_weights['xgb'] * xgb_importance
        )
        
        feature_importance = dict(zip(self.feature_names, weighted_importance))
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, filepath: str = "models/fraud_detector.pkl"):
        """Save the trained model."""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_weights': self.model_weights,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath: str = "models/fraud_detector.pkl"):
        """Load a trained model."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.rf_model = model_data['rf_model']
        self.xgb_model = model_data['xgb_model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_weights = model_data['model_weights']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Model loaded from: {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        return {
            "status": "Trained",
            "feature_count": len(self.feature_names),
            "model_weights": self.model_weights,
            "feature_names": self.feature_names
        }
