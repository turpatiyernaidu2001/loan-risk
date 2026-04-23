import shap
import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
import pandas as pd
import joblib
import os
from utils.preprocess import preprocess_data

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
model = joblib.load(model_path)
explainer = shap.TreeExplainer(model)

# Load columns (PLACE HERE)
columns_path = os.path.join(os.path.dirname(__file__), "..", "model", "columns.pkl")
model_columns = joblib.load(columns_path)

importance_path = os.path.join(os.path.dirname(__file__), "..", "model", "feature_importance.pkl")
feature_importance = joblib.load(importance_path)

@app.get("/")
def home():
    return {"message": "Loan Risk API is running"}


# Decision logic
def loan_decision(score):
    if score < 30:
        return "Approve"
    elif score < 50:
        return "Conditional Approve"
    elif score < 70:
        return "Review"
    else:
        return "Reject"

def get_risk_tier(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    else:
        return "High"

feature_name_map = {
    "person_income": "Income",
    "loan_amnt": "Loan amount",
    "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan vs income ratio",
    "cb_person_cred_hist_length": "Credit history length",
    "person_emp_length": "Employment length",

    # Encoded features
    "person_home_ownership_RENT": "Rental housing",
    "person_home_ownership_OWN": "Own house",
    "loan_intent_PERSONAL": "Personal loan",
    "loan_intent_EDUCATION": "Education loan",
    "loan_intent_HOMEIMPROVEMENT": "Home improvement loan",
}
def get_shap_explanations(df, data):
    try:
        import numpy as np

        shap_values = explainer.shap_values(df)

        if isinstance(shap_values, list):
            values = shap_values[1]
        else:
            values = shap_values

        values = np.array(values).flatten()

        features = df.columns
        shap_data = list(zip(features, values))

        shap_data = sorted(shap_data, key=lambda x: abs(x[1]), reverse=True)

        explanations = []

        for feature, value in shap_data:
            # ✅ Loan Grade (SMART)
            if feature == "loan_grade":
                grade = data.get("loan_grade", "C")

                if grade == "A":
                    explanations.append("Excellent credit grade — very low risk")
                elif grade == "B":
                    explanations.append("Good credit grade — low risk")
                elif grade == "C":
                    explanations.append("Moderate credit grade — balanced risk")
                elif grade == "D":
                    explanations.append("Below-average credit grade — higher risk")
                else:
                    explanations.append("Poor credit grade — high default risk")
                continue

            # ✅ Loan vs Income
            if feature == "loan_percent_income":
                ratio = data.get("loan_percent_income", 0)

                if ratio > 0.5:
                    explanations.append(f"Loan is {round(ratio*100)}% of income — very high financial burden")
                elif ratio > 0.3:
                    explanations.append(f"Loan is {round(ratio*100)}% of income — moderate risk")
                else:
                    explanations.append(f"Loan is {round(ratio*100)}% of income — manageable level")
                continue

            # ✅ Income
            if feature == "person_income":
                income = data.get("person_income", 0)

                if income < 30000:
                    explanations.append("Low income reduces repayment ability")
                elif income < 70000:
                    explanations.append("Moderate income supports repayment")
                else:
                    explanations.append("High income improves repayment capacity")
                continue

            # ✅ Interest Rate
            if feature == "loan_int_rate":
                rate = data.get("loan_int_rate", 0)

                if rate > 15:
                    explanations.append(f"High interest rate ({rate}%) increases repayment burden")
                elif rate > 10:
                    explanations.append(f"Moderate interest rate ({rate}%)")
                else:
                    explanations.append(f"Low interest rate ({rate}%) reduces risk")
                continue

            # ✅ Credit History
            if feature == "cb_person_cred_hist_length":
                hist = data.get("cb_person_cred_hist_length", 0)

                if hist < 3:
                    explanations.append("Short credit history increases uncertainty")
                elif hist < 7:
                    explanations.append("Moderate credit history")
                else:
                    explanations.append("Long credit history improves reliability")
                continue

            # ✅ Default History
            if feature == "cb_person_default_on_file":
                if data.get("cb_person_default_on_file") == "N":
                    explanations.append("No previous loan default — positive signal")
                else:
                    explanations.append("Previous loan default — high risk")
                continue

            # ✅ Loan Intent
            if feature.startswith("loan_intent_"):
                if feature != f"loan_intent_{data.get('loan_intent')}":
                    continue

                intent = data.get("loan_intent")
                explanations.append(f"Loan purpose: {intent.lower()}")
                continue

            # ✅ Home Ownership
            if feature.startswith("person_home_ownership_"):
                if feature != f"person_home_ownership_{data.get('person_home_ownership')}":
                    continue

                home = data.get("person_home_ownership")

                if home == "RENT":
                    explanations.append("Rental housing increases monthly financial pressure")
                else:
                    explanations.append("Home ownership provides financial stability")
                continue

            # ✅ Fallback
            feature_name = feature_name_map.get(feature, feature)

            if value > 0:
                explanations.append(f"{feature_name} increases risk")
            else:
                explanations.append(f"{feature_name} reduces risk")

            if len(explanations) == 5:
                break

        # ✅ RETURN MUST BE HERE
        if len(explanations) == 0:
            return ["No strong risk factors detected"]

        return explanations

    except Exception as e:
        print("SHAP ERROR:", e)
        return ["Explanation not available"]

def log_prediction(input_data, risk_score, risk_tier, decision):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ Define exact column order
    columns = [
        "person_age",
        "person_income",
        "person_home_ownership",
        "person_emp_length",
        "loan_intent",
        "loan_grade",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_default_on_file",
        "cb_person_cred_hist_length",
        "risk_score",
        "risk_tier",   # ✅ correct position
        "decision",    # ✅ correct position
        "timestamp"
    ]

    # ✅ Create row in correct order
    row = [
        input_data.get("person_age"),
        input_data.get("person_income"),
        input_data.get("person_home_ownership"),
        input_data.get("person_emp_length"),
        input_data.get("loan_intent"),
        input_data.get("loan_grade"),
        input_data.get("loan_amnt"),
        input_data.get("loan_int_rate"),
        input_data.get("loan_percent_income"),
        input_data.get("cb_person_default_on_file"),
        input_data.get("cb_person_cred_hist_length"),
        risk_score,
        risk_tier,   # ✅ FIRST
        decision,    # ✅ SECOND
        timestamp
    ]

    log_df = pd.DataFrame([row], columns=columns)

    log_path = os.path.join(os.path.dirname(__file__), "..", "data", "logs.csv")

    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)
 

@app.post("/predict")
def predict(data: dict):

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Preprocess
    df = preprocess_data(df)
    df = df.reindex(columns=model_columns, fill_value=0)

    df = df.astype(float)


    # Predict
    prob = model.predict_proba(df)[:, 1][0]
    risk_score = prob * 100
    confidence = abs(prob - 0.5) * 200

    risk_tier = get_risk_tier(risk_score)
    decision = loan_decision(risk_score)
    explanations = get_shap_explanations(df,data)

    log_prediction(data, risk_score, risk_tier, decision)

    return {
        "input_data": data,
        "risk_score": round(risk_score, 2),
        "confidence": round(confidence, 2),
        "risk_tier": risk_tier,
        "decision": decision,
        "explanations": explanations
    }