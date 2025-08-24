"""
Credit Card Fraud Detection - Data Generation Module

This module generates synthetic credit card transaction data with realistic
fraud patterns for training and testing the fraud detection model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, List
import os

class FraudDataGenerator:
    """Generates synthetic credit card transaction data with fraud patterns."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed."""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Merchant categories with different fraud rates
        self.merchant_categories = {
            'online_retail': 0.15,      # Higher fraud rate
            'gas_station': 0.05,        # Lower fraud rate
            'grocery_store': 0.03,      # Very low fraud rate
            'restaurant': 0.04,         # Low fraud rate
            'electronics': 0.12,        # Higher fraud rate
            'travel': 0.08,             # Medium fraud rate
            'pharmacy': 0.02,           # Very low fraud rate
            'jewelry': 0.18,            # Very high fraud rate
            'entertainment': 0.06,      # Medium fraud rate
            'utilities': 0.01           # Very low fraud rate
        }
        
        # Fraud patterns
        self.fraud_patterns = {
            'high_amount': 0.3,         # 30% of frauds are high amounts
            'unusual_time': 0.25,       # 25% of frauds are at unusual times
            'far_location': 0.35,       # 35% of frauds are far from home
            'online_order': 0.4,        # 40% of frauds are online
            'new_merchant': 0.45        # 45% of frauds are at new merchants
        }
    
    def generate_transaction_features(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic transaction features."""
        
        # Basic transaction features
        data = {
            'amount': np.random.lognormal(mean=4.5, sigma=0.8, size=n_samples),
            'merchant_category': np.random.choice(
                list(self.merchant_categories.keys()), 
                size=n_samples
            ),
            'hour': np.random.randint(0, 24, size=n_samples),
            'day_of_week': np.random.randint(0, 7, size=n_samples),
            'distance_from_home': np.random.exponential(scale=10, size=n_samples),
            'distance_from_last_transaction': np.random.exponential(scale=5, size=n_samples),
            'ratio_to_median_purchase_price': np.random.lognormal(mean=0, sigma=0.5, size=n_samples),
            'repeat_retailer': np.random.choice([True, False], size=n_samples, p=[0.3, 0.7]),
            'used_chip': np.random.choice([True, False], size=n_samples, p=[0.8, 0.2]),
            'used_pin_number': np.random.choice([True, False], size=n_samples, p=[0.6, 0.4]),
            'online_order': np.random.choice([True, False], size=n_samples, p=[0.4, 0.6])
        }
        
        return pd.DataFrame(data)
    
    def generate_fraud_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate fraud labels based on realistic patterns."""
        
        n_samples = len(df)
        fraud_labels = np.zeros(n_samples, dtype=int)
        
        # Base fraud rates by merchant category
        for category, fraud_rate in self.merchant_categories.items():
            category_mask = df['merchant_category'] == category
            n_category = category_mask.sum()
            n_fraud = int(n_category * fraud_rate)
            
            if n_fraud > 0:
                fraud_indices = np.random.choice(
                    np.where(category_mask)[0], 
                    size=n_fraud, 
                    replace=False
                )
                fraud_labels[fraud_indices] = 1
        
        # Apply fraud patterns to increase realism
        fraud_indices = np.where(fraud_labels == 1)[0]
        
        for fraud_idx in fraud_indices:
            # High amount fraud
            if np.random.random() < self.fraud_patterns['high_amount']:
                df.loc[fraud_idx, 'amount'] *= np.random.uniform(3, 10)
                df.loc[fraud_idx, 'ratio_to_median_purchase_price'] *= np.random.uniform(2, 5)
            
            # Unusual time fraud
            if np.random.random() < self.fraud_patterns['unusual_time']:
                df.loc[fraud_idx, 'hour'] = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23])
            
            # Far location fraud
            if np.random.random() < self.fraud_patterns['far_location']:
                df.loc[fraud_idx, 'distance_from_home'] *= np.random.uniform(3, 8)
                df.loc[fraud_idx, 'distance_from_last_transaction'] *= np.random.uniform(2, 6)
            
            # Online order fraud
            if np.random.random() < self.fraud_patterns['online_order']:
                df.loc[fraud_idx, 'online_order'] = True
            
            # New merchant fraud
            if np.random.random() < self.fraud_patterns['new_merchant']:
                df.loc[fraud_idx, 'repeat_retailer'] = False
        
        return fraud_labels
    
    def generate_dataset(self, n_samples: int = 100000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate complete dataset with features and labels."""
        
        print(f"Generating {n_samples:,} transaction records...")
        
        # Generate features
        df = self.generate_transaction_features(n_samples)
        
        # Generate fraud labels
        fraud_labels = self.generate_fraud_labels(df)
        
        # Add fraud column
        df['fraud'] = fraud_labels
        
        # Calculate fraud statistics
        fraud_rate = fraud_labels.mean()
        print(f"Generated dataset with {fraud_rate:.2%} fraud rate")
        print(f"Total fraud cases: {fraud_labels.sum():,}")
        print(f"Total legitimate cases: {(fraud_labels == 0).sum():,}")
        
        return df, fraud_labels
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "credit_card_transactions.csv"):
        """Save the generated dataset to CSV file."""
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")
        
        return filepath

def main():
    """Main function to generate and save the dataset."""
    
    print("🚀 Starting Credit Card Fraud Data Generation...")
    
    # Initialize generator
    generator = FraudDataGenerator(seed=42)
    
    # Generate dataset
    df, labels = generator.generate_dataset(n_samples=100000)
    
    # Save dataset
    filepath = generator.save_dataset(df)
    
    # Display dataset info
    print("\n📊 Dataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")  # Exclude 'fraud' column
    print(f"Target: fraud")
    
    print("\n📈 Fraud Distribution by Merchant Category:")
    fraud_by_category = df.groupby('merchant_category')['fraud'].agg(['count', 'sum', 'mean'])
    fraud_by_category.columns = ['Total', 'Fraud', 'Fraud_Rate']
    print(fraud_by_category.sort_values('Fraud_Rate', ascending=False))
    
    print("\n✅ Data generation completed successfully!")
    
    return df, labels

if __name__ == "__main__":
    main()
