import requests
import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
API_URL = "https://loan-risk-api-2jne.onrender.com/predict"
st.title("📊 Loan Risk Portfolio Dashboard")

# 🔹 Load metrics
metrics_path = os.path.join(os.path.dirname(__file__), "..", "model", "metrics.json")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.subheader("📈 Model Performance")
    st.write(f"Accuracy: {metrics['accuracy']}")
    st.write(f"Last Trained: {metrics['timestamp']}")

# 🔹 Load logs
log_path = os.path.join(os.path.dirname(__file__), "..", "data", "logs.csv")

if not os.path.exists(log_path): 
    st.warning("No data available yet.")
else:
    df = pd.read_csv(log_path)

    # ✅ Clean decision column (VERY IMPORTANT)
    valid_decisions = ["Approve", "Conditional Approve", "Review", "Reject"]
    df = df[df["decision"].isin(valid_decisions)]

    valid_tiers = ["Low", "Medium", "High"]
    df = df[df["risk_tier"].isin(valid_tiers)]

    # 🔍 DEBUG (optional, remove later)
    # st.write(df.head())

    # 🔹 Total applications
    st.subheader("Total Applications")
    st.write(len(df))

    # 🔹 Average Risk Score
    st.subheader("Average Risk Score")
    st.write(round(df["risk_score"].mean(), 2))

    # 🔹 Risk Tier Distribution
    st.subheader("Risk Tier Distribution")
    st.bar_chart(df["risk_tier"].value_counts())

    st.subheader("Risk Tier Distribution (Pie Chart)")

    risk_counts = df["risk_tier"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%')

    st.pyplot(fig1)    

    # 🔹 Decision Distribution
    st.subheader("Decision Distribution")
    st.bar_chart(df["decision"].value_counts())

    st.subheader("Decision Distribution (Pie Chart)")

    decision_counts = df["decision"].value_counts()

    fig2, ax2 = plt.subplots()
    ax2.pie(decision_counts, labels=decision_counts.index, autopct='%1.1f%%')

    st.pyplot(fig2)

    # 🔍 Filter
    st.subheader("🔍 Filter by Risk Tier")

    selected_tier = st.selectbox(
        "Select Risk Tier",
        ["All"] + list(df["risk_tier"].unique())
    )

    if selected_tier != "All":
        filtered_df = df[df["risk_tier"] == selected_tier]
    else:
        filtered_df = df

    # 🔹 Table
    st.subheader("Recent Applications")
    st.dataframe(filtered_df)


st.subheader("🧪 Test Model API")

income = st.number_input("Income")
loan = st.number_input("Loan Amount")

if st.button("Predict"):
    payload = {
        "income": income,
        "loan_amount": loan
    }

    response = requests.post(API_URL, json=payload)

    st.write("🔍 Raw Response:", response.text)  # ADD THIS DEBUG LINE

    if response.status_code == 200:
        st.success(response.json())
    else:
        st.error(f"API Error: {response.text}")