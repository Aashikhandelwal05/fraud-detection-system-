"""
Unit tests for the fraud detection model.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.fraud_detector import FraudDetector
from data.generate_data import FraudDataGenerator

class TestFraudDetector:
    """Test cases for FraudDetector class."""
    
    @pytest.fixture
    def fraud_detector(self):
        """Create a FraudDetector instance for testing."""
        return FraudDetector(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        generator = FraudDataGenerator(seed=42)
        df, labels = generator.generate_dataset(n_samples=1000)
        return df
    
    def test_initialization(self, fraud_detector):
        """Test FraudDetector initialization."""
        assert fraud_detector.random_state == 42
        assert fraud_detector.is_trained == False
        assert fraud_detector.feature_names is None
        assert fraud_detector.model_weights == {'rf': 0.5, 'xgb': 0.5}
    
    def test_preprocess_features(self, fraud_detector, sample_data):
        """Test feature preprocessing."""
        # Remove target column for preprocessing
        df_features = sample_data.drop('fraud', axis=1)
        
        processed_df = fraud_detector.preprocess_features(df_features)
        
        # Check that categorical variables are encoded
        assert 'merchant_category' not in processed_df.columns
        assert 'merchant_category_encoded' in processed_df.columns
        
        # Check that boolean columns are converted to int
        boolean_columns = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
        for col in boolean_columns:
            assert processed_df[col].dtype in [np.int32, np.int64]
        
        # Check that engineered features are added
        engineered_features = ['is_night', 'is_weekend', 'amount_log', 'amount_squared']
        for feature in engineered_features:
            assert feature in processed_df.columns
    
    def test_prepare_data(self, fraud_detector, sample_data):
        """Test data preparation."""
        X, y = fraud_detector.prepare_data(sample_data, target_col='fraud')
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(sample_data.columns) - 1  # Exclude target column
        assert fraud_detector.feature_names is not None
    
    def test_model_training(self, fraud_detector, sample_data):
        """Test model training."""
        results = fraud_detector.train(sample_data, target_col='fraud', test_size=0.2, use_smote=True)
        
        # Check that model is trained
        assert fraud_detector.is_trained == True
        
        # Check that results contain expected keys
        assert 'metrics' in results
        assert 'feature_importance' in results
        assert 'model_weights' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        
        # Check model weights
        weights = results['model_weights']
        assert 'rf' in weights
        assert 'xgb' in weights
        assert abs(weights['rf'] + weights['xgb'] - 1.0) < 1e-6
    
    def test_prediction(self, fraud_detector, sample_data):
        """Test model prediction."""
        # Train the model first
        fraud_detector.train(sample_data, target_col='fraud', test_size=0.2, use_smote=True)
        
        # Prepare test data
        df_features = sample_data.drop('fraud', axis=1)
        processed_df = fraud_detector.preprocess_features(df_features)
        X_test = processed_df.values[:10]  # Test with first 10 samples
        
        # Test probability prediction
        proba = fraud_detector.predict_proba(X_test)
        assert isinstance(proba, np.ndarray)
        assert proba.shape[0] == 10
        assert all(0 <= p <= 1 for p in proba)
        
        # Test label prediction
        predictions = fraud_detector.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape[0] == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_save_load(self, fraud_detector, sample_data, tmp_path):
        """Test model saving and loading."""
        # Train the model
        fraud_detector.train(sample_data, target_col='fraud', test_size=0.2, use_smote=True)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        fraud_detector.save_model(str(model_path))
        
        # Create new instance and load model
        new_detector = FraudDetector()
        new_detector.load_model(str(model_path))
        
        # Check that loaded model is trained
        assert new_detector.is_trained == True
        assert new_detector.feature_names == fraud_detector.feature_names
        assert new_detector.model_weights == fraud_detector.model_weights
        
        # Test prediction with loaded model
        df_features = sample_data.drop('fraud', axis=1)
        processed_df = new_detector.preprocess_features(df_features)
        X_test = processed_df.values[:5]
        
        original_proba = fraud_detector.predict_proba(X_test)
        loaded_proba = new_detector.predict_proba(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_proba, loaded_proba, decimal=5)
    
    def test_model_info(self, fraud_detector, sample_data):
        """Test model info retrieval."""
        # Before training
        info = fraud_detector.get_model_info()
        assert info["status"] == "Model not trained"
        
        # After training
        fraud_detector.train(sample_data, target_col='fraud', test_size=0.2, use_smote=True)
        info = fraud_detector.get_model_info()
        
        assert info["status"] == "Trained"
        assert "feature_count" in info
        assert "model_weights" in info
        assert "feature_names" in info
    
    def test_invalid_prediction_before_training(self, fraud_detector, sample_data):
        """Test that prediction fails before training."""
        df_features = sample_data.drop('fraud', axis=1)
        processed_df = fraud_detector.preprocess_features(df_features)
        X_test = processed_df.values[:5]
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            fraud_detector.predict_proba(X_test)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            fraud_detector.predict(X_test)
    
    def test_invalid_save_before_training(self, fraud_detector, tmp_path):
        """Test that saving fails before training."""
        model_path = tmp_path / "test_model.pkl"
        
        with pytest.raises(ValueError, match="Model must be trained before saving"):
            fraud_detector.save_model(str(model_path))

class TestFraudDataGenerator:
    """Test cases for FraudDataGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a FraudDataGenerator instance for testing."""
        return FraudDataGenerator(seed=42)
    
    def test_initialization(self, generator):
        """Test FraudDataGenerator initialization."""
        assert generator.seed == 42
        assert len(generator.merchant_categories) > 0
        assert len(generator.fraud_patterns) > 0
    
    def test_generate_transaction_features(self, generator):
        """Test transaction feature generation."""
        n_samples = 100
        df = generator.generate_transaction_features(n_samples)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_samples
        assert 'amount' in df.columns
        assert 'merchant_category' in df.columns
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
    
    def test_generate_dataset(self, generator):
        """Test complete dataset generation."""
        n_samples = 1000
        df, labels = generator.generate_dataset(n_samples)
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(labels, np.ndarray)
        assert len(df) == n_samples
        assert len(labels) == n_samples
        assert 'fraud' in df.columns
        assert all(label in [0, 1] for label in labels)
        
        # Check fraud rate is reasonable
        fraud_rate = labels.mean()
        assert 0.01 <= fraud_rate <= 0.20  # Between 1% and 20%
    
    def test_save_dataset(self, generator, tmp_path):
        """Test dataset saving."""
        df, labels = generator.generate_dataset(100)
        
        filepath = tmp_path / "test_data.csv"
        saved_path = generator.save_dataset(df, str(filepath))
        
        assert saved_path == str(filepath)
        assert filepath.exists()
        
        # Load and verify
        loaded_df = pd.read_csv(filepath)
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
