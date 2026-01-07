import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Retail Intelligence Dashboard", layout="wide")
st.title("Retail Churn Prediction Dashboard")

st.sidebar.header("Customer Input Metrics")
recency = st.sidebar.number_input("Recency (Days since last purchase)", value=30)
frequency = st.sidebar.number_input("Frequency (Number of Invoices)", value=5)
monetary = st.sidebar.number_input("Monetary (Total Spend)", value=1000.0)
cluster = st.sidebar.selectbox("Customer Segment (Cluster)", options=[0, 1, 2, 3])

if st.sidebar.button("Predict Churn"):
    payload = {
        "Recency": recency,
        "Frequency": int(frequency),
        "Monetary": monetary,
        "Cluster": int(cluster)
    }
    
    # Connect to the FastAPI service using the Docker service name
    response = requests.post("http://api:8000/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        color = "red" if result["churn_prediction"] == "Yes" else "green"
        st.markdown(f"### Prediction: :{color}[{result['churn_prediction']}]")
        st.metric("Churn Probability", f"{result['churn_probability'] * 100:.2f}%")
    else:
        st.error("Error connecting to the API.")
