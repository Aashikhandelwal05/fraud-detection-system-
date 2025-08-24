"""
Credit Card Fraud Detection - Streamlit Dashboard

Real-time monitoring dashboard for fraud detection system with
interactive visualizations and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .success-card {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

class FraudDetectionDashboard:
    """Main dashboard class for fraud detection monitoring."""
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.session = requests.Session()
        
    def check_api_health(self) -> bool:
        """Check if the API is running."""
        try:
            response = self.session.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_api_metrics(self) -> Optional[Dict]:
        """Get API performance metrics."""
        try:
            response = self.session.get(f"{self.api_base_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_model_info(self) -> Optional[Dict]:
        """Get model information."""
        try:
            response = self.session.get(f"{self.api_base_url}/model/info", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def predict_fraud(self, transaction_data: Dict) -> Optional[Dict]:
        """Make fraud prediction."""
        try:
            response = self.session.post(
                f"{self.api_base_url}/predict",
                json=transaction_data,
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
        return None

def create_header():
    """Create the main header."""
    st.markdown('<h1 class="main-header">🛡️ Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

def create_sidebar():
    """Create the sidebar with configuration options."""
    st.sidebar.title("⚙️ Configuration")
    
    # API Status
    api_status = dashboard.check_api_health()
    if api_status:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
    
    # Refresh rate
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 30, 5)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    
    st.sidebar.markdown("---")
    
    # Model Information
    st.sidebar.subheader("🤖 Model Info")
    model_info = dashboard.get_model_info()
    if model_info:
        st.sidebar.metric("Features", model_info.get("feature_count", 0))
        st.sidebar.metric("Status", "Trained")
    else:
        st.sidebar.metric("Status", "Not Available")
    
    return refresh_rate, auto_refresh

def create_overview_metrics(metrics: Dict):
    """Create overview metrics cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Predictions",
            f"{metrics.get('total_predictions', 0):,}",
            delta=None
        )
    
    with col2:
        fraud_rate = metrics.get('fraud_rate', 0)
        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.2%}",
            delta=None
        )
    
    with col3:
        avg_response = metrics.get('avg_response_time', 0)
        st.metric(
            "Avg Response Time",
            f"{avg_response:.1f}ms",
            delta=None
        )
    
    with col4:
        fraud_predictions = metrics.get('fraud_predictions', 0)
        st.metric(
            "Fraud Detected",
            f"{fraud_predictions:,}",
            delta=None
        )

def create_performance_charts(metrics: Dict):
    """Create performance visualization charts."""
    st.subheader("📊 Performance Analytics")
    
    # Get recent predictions
    recent_predictions = metrics.get('recent_predictions', [])
    
    if recent_predictions:
        # Convert to DataFrame
        df = pd.DataFrame(recent_predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Probability Over Time', 'Transaction Amounts', 
                          'Response Times', 'Fraud vs Legitimate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Fraud probability over time
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['fraud_probability'], 
                      mode='lines+markers', name='Fraud Probability'),
            row=1, col=1
        )
        
        # Transaction amounts
        fig.add_trace(
            go.Bar(x=df['transaction_id'], y=df['amount'], 
                  name='Amount', marker_color='lightblue'),
            row=1, col=2
        )
        
        # Response times
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['response_time_ms'], 
                      mode='lines+markers', name='Response Time (ms)'),
            row=2, col=1
        )
        
        # Fraud vs legitimate pie chart
        fraud_count = df['is_fraud'].sum()
        legitimate_count = len(df) - fraud_count
        
        fig.add_trace(
            go.Pie(labels=['Legitimate', 'Fraud'], 
                  values=[legitimate_count, fraud_count],
                  marker_colors=['lightgreen', 'lightcoral']),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent predictions available for visualization.")

def create_real_time_prediction():
    """Create real-time prediction interface."""
    st.subheader("🔮 Real-time Fraud Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.01, value=100.0, step=0.01)
            merchant_category = st.selectbox(
                "Merchant Category",
                ['online_retail', 'gas_station', 'grocery_store', 'restaurant', 
                 'electronics', 'travel', 'pharmacy', 'jewelry', 'entertainment', 'utilities']
            )
            hour = st.slider("Hour", 0, 23, 12)
            day_of_week = st.slider("Day of Week", 0, 6, 2)
            distance_from_home = st.number_input("Distance from Home (miles)", min_value=0.0, value=10.0, step=0.1)
        
        with col2:
            distance_from_last_transaction = st.number_input("Distance from Last Transaction (miles)", min_value=0.0, value=5.0, step=0.1)
            ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.1, value=1.0, step=0.1)
            repeat_retailer = st.checkbox("Repeat Retailer")
            used_chip = st.checkbox("Used Chip")
            used_pin_number = st.checkbox("Used PIN Number")
            online_order = st.checkbox("Online Order")
        
        submitted = st.form_submit_button("Predict Fraud")
        
        if submitted:
            # Prepare transaction data
            transaction_data = {
                "amount": amount,
                "merchant_category": merchant_category,
                "hour": hour,
                "day_of_week": day_of_week,
                "distance_from_home": distance_from_home,
                "distance_from_last_transaction": distance_from_last_transaction,
                "ratio_to_median_purchase_price": ratio_to_median_purchase_price,
                "repeat_retailer": repeat_retailer,
                "used_chip": used_chip,
                "used_pin_number": used_pin_number,
                "online_order": online_order
            }
            
            # Make prediction
            with st.spinner("Analyzing transaction..."):
                prediction = dashboard.predict_fraud(transaction_data)
            
            if prediction:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    fraud_prob = prediction['fraud_probability']
                    is_fraud = prediction['is_fraud']
                    
                    if is_fraud:
                        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                        st.error(f"🚨 FRAUD DETECTED!")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        st.success(f"✅ LEGITIMATE TRANSACTION")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                    st.metric("Confidence", prediction['confidence'])
                    st.metric("Response Time", f"{prediction['response_time_ms']:.1f}ms")
                
                with col2:
                    st.subheader("Risk Factors")
                    risk_factors = prediction['risk_factors']
                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(f"⚠️ {factor}")
                    else:
                        st.info("✅ No significant risk factors identified")
                    
                    st.subheader("Transaction Details")
                    st.write(f"**Transaction ID:** {prediction['transaction_id']}")
                    st.write(f"**Timestamp:** {prediction['timestamp']}")

def create_batch_prediction():
    """Create batch prediction interface."""
    st.subheader("📦 Batch Prediction")
    
    # Sample transactions
    sample_transactions = [
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
        }
    ]
    
    uploaded_file = st.file_uploader("Upload CSV file with transactions", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Process Batch"):
                # Convert DataFrame to list of dictionaries
                transactions = df.to_dict('records')
                
                with st.spinner("Processing batch predictions..."):
                    # Make batch prediction
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/predict/batch",
                            json=transactions,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            st.success(f"✅ Processed {results['total_transactions']} transactions")
                            st.metric("Total Time", f"{results['total_time_ms']:.1f}ms")
                            st.metric("Avg Time per Transaction", f"{results['avg_time_per_transaction_ms']:.1f}ms")
                            
                            # Display results
                            results_df = pd.DataFrame(results['predictions'])
                            st.dataframe(results_df)
                        else:
                            st.error("Failed to process batch")
                    except Exception as e:
                        st.error(f"Batch processing error: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Quick test with sample data
    if st.button("Test with Sample Data"):
        with st.spinner("Processing sample transactions..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict/batch",
                    json=sample_transactions,
                    timeout=30
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    st.success(f"✅ Processed {results['total_transactions']} sample transactions")
                    
                    # Display results
                    results_df = pd.DataFrame(results['predictions'])
                    st.dataframe(results_df)
                else:
                    st.error("Failed to process sample data")
            except Exception as e:
                st.error(f"Sample processing error: {str(e)}")

def main():
    """Main dashboard function."""
    global dashboard
    
    # Initialize dashboard
    dashboard = FraudDetectionDashboard()
    
    # Create header
    create_header()
    
    # Create sidebar
    refresh_rate, auto_refresh = create_sidebar()
    
    # Check API health
    if not dashboard.check_api_health():
        st.error("❌ Cannot connect to Fraud Detection API. Please ensure the API is running on http://localhost:8000")
        st.info("To start the API, run: `uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000`")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Real-time Prediction", "📦 Batch Prediction", "📈 Analytics"])
    
    with tab1:
        # Overview metrics
        metrics = dashboard.get_api_metrics()
        if metrics:
            create_overview_metrics(metrics)
            
            # Performance charts
            create_performance_charts(metrics)
        else:
            st.warning("Unable to fetch metrics from API")
    
    with tab2:
        create_real_time_prediction()
    
    with tab3:
        create_batch_prediction()
    
    with tab4:
        st.subheader("📈 Advanced Analytics")
        
        # Model performance over time
        st.write("Model performance metrics and trends will be displayed here.")
        
        # Feature importance
        model_info = dashboard.get_model_info()
        if model_info and 'top_features' in model_info:
            st.subheader("🔝 Top Features")
            features = model_info['top_features']
            if features:
                feature_df = pd.DataFrame({
                    'Feature': features,
                    'Rank': range(1, len(features) + 1)
                })
                st.dataframe(feature_df)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()
