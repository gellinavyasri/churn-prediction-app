import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define a dummy fallback class if needed by joblib
class Dummy:
    def __init__(*args, **kwargs): pass

# Fix for loading custom class-based models
import cloudpickle

def custom_joblib_load(filename):
    with open(filename, 'rb') as f:
        return cloudpickle.load(f)

model = custom_joblib_load('churn_model_compressed (2).joblib')


model = custom_joblib_load('churn_model_compressed (2).joblib')

st.title("Customer Churn Prediction")
st.write("""
This app predicts whether a customer is likely to churn or not.

👉 **Please enter the details below:**
""")

# User input
tenure = st.number_input('Tenure (in months)', min_value=0, max_value=100)
monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0)
contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

if st.button('Predict'):
    input_data = pd.DataFrame({
        'Tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract': [contract],
        'InternetService': [internet_service],
        'PaymentMethod': [payment_method],
    })

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error(f"⚠️ The customer is likely to **churn**. (Confidence: {prediction_proba[0][1]:.2f})")
    else:
        st.success(f"✅ The customer is **not likely to churn**. (Confidence: {prediction_proba[0][0]:.2f})")
