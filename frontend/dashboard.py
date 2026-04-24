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


st.subheader("🧪 Loan Application Form")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)

person_income = st.number_input("Monthly Income", value=50000)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

person_emp_length = st.number_input("Employment Length (years)", value=2)

loan_intent = st.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "VENTURE", "DEBTCONSOLIDATION"]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input("Loan Amount", value=10000)

loan_int_rate = st.number_input("Interest Rate", value=10.0)

loan_percent_income = loan_amnt / person_income if person_income != 0 else 0

cb_person_default_on_file = st.selectbox(
    "Previous Default?",
    ["N", "Y"]
)

cb_person_cred_hist_length = st.number_input("Credit History Length", value=5)


if st.button("Predict"):
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }

    response = requests.post(API_URL, json=payload)

    st.write("🔍 Raw Response:", response.text)

    if response.status_code == 200:
        result = response.json()

        st.success(f"Decision: {result['decision']}")
        st.info(f"Risk Score: {result['risk_score']}")
        st.info(f"Confidence: {result['confidence']}")
        st.warning(f"Risk Tier: {result['risk_tier']}")
        

        st.subheader("📌 Explanations")
        for e in result["explanations"]:
            st.write("•", e)
    else:
        st.error("API Error")