import streamlit as st
import requests

st.title("Loan Risk Prediction System")

st.write("Enter applicant details:")

# Input fields
person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
person_income = st.number_input("Income", value=50000)
person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.number_input("Employment Length (years)", value=5)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.number_input("Loan Amount", value=10000)
loan_int_rate = st.number_input("Interest Rate", value=12.5)
loan_percent_income = st.number_input("Loan % Income", value=0.2)
cb_person_default_on_file = st.selectbox("Previous Default", ["Y", "N"])
cb_person_cred_hist_length = st.number_input("Credit History Length", value=3)

# Button
if st.button("Predict Risk"):

    data = {
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

    # Call FastAPI
    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Risk Score: {result['risk_score']}")
        st.success(f"Decision: {result['decision']}")
    else:
        st.error("Error connecting to API")