import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('churn_model_compressed.joblib')

# Streamlit UI
st.title("📊 Customer Churn Prediction App")
st.markdown("Enter Customer Details 👇")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (in months)", 0, 72, 12)
usage_freq = st.slider("Usage Frequency (times/month)", 0, 30, 10)
support_calls = st.slider("Support Calls", 0, 20, 5)
payment_delay = st.slider("Payment Delay (days)", 0, 30, 5)
subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend = st.number_input("Total Spend ($)", 0.0, 5000.0, 100.0)
last_interaction = st.slider("Last Interaction (days ago)", 0, 30, 10)

# Create input DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Tenure': [tenure],
    'Usage Frequency': [usage_freq],
    'Support Calls': [support_calls],
    'Payment Delay': [payment_delay],
    'Subscription Type': [subscription_type],
    'Contract Length': [contract_length],
    'Total Spend': [total_spend],
    'Last Interaction': [last_interaction]
})

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    st.write("### 🔍 Prediction Result:")
    st.success("Customer is likely to churn." if prediction == 1 else "Customer is likely to stay.")
    st.write(f"📈 Churn Probability: **{prob:.2f}**")
