import streamlit as st
import pandas as pd
import numpy as np
import joblib

import gdown
import os

# Download the model from Google Drive only if not already downloaded
model_path = 'churn_model_compressed.joblib'
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=1cBzxnm0WTqsZKfrTc4ps0ndvvsIQwttr'
    gdown.download(url, model_path, quiet=False)

# Now load the model
import joblib
model = joblib.load(model_path)

st.title("Customer Churn Prediction")
st.write("""
This app predicts whether a customer is likely to churn or not.

👉 **Please enter the details below:**
""")

# Example user input form — You can adjust based on your actual features
tenure = st.number_input('Tenure (in months)', min_value=0, max_value=100)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0)
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

# Add other required inputs as per your model's features...

if st.button('Predict'):
    # Create a DataFrame with one row
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'InternetService': [internet_service],
        'PaymentMethod': [payment_method],
        # Add all other required columns here...
    })

    # Predict
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to **churn**. (Confidence: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"✅ The customer is **not likely to churn**. (Confidence: {prediction_proba[0][0]:.2f})")
