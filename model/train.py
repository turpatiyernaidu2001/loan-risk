import pandas as pd
import os
from utils.preprocess import preprocess_data

# Load data
df = pd.read_csv(r"D:\projects\loanrisk project\data\credit_risk_dataset.csv")

# Apply preprocessing
df = preprocess_data(df)

# Split
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
from datetime import datetime
import json

metrics = {
    "accuracy": float(accuracy),
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
current_dir = os.path.dirname(__file__)
metrics_path = os.path.join(current_dir, "metrics.json")

with open(metrics_path, "w") as f:
    json.dump(metrics, f)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
import pandas as pd
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

risk_score = y_prob * 100

def loan_decision(score):
    if score < 30:
        return "Approve"
    elif score < 50:
        return "Conditional Approve"
    elif score < 70:
        return "Review"
    else:
        return "Reject"
        
import pandas as pd

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Risk Score": risk_score
})

results["Decision"] = results["Risk Score"].apply(loan_decision)

print("\nDecision Distribution:")
print(results["Decision"].value_counts())

print("\nRisk Score Distribution:")
print(results["Risk Score"].describe())


print("\nSample Results:")
print(results.head())


import os
import joblib

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "model.pkl")

joblib.dump(model, file_path)

import joblib
import os

# Save model
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "model.pkl")
joblib.dump(model, model_path)

# Save columns
columns_path = os.path.join(current_dir, "columns.pkl")
joblib.dump(X.columns.tolist(), columns_path)

import os
import joblib

current_dir = os.path.dirname(__file__)

# Save feature importance
importance_path = os.path.join(current_dir, "feature_importance.pkl")
joblib.dump(feature_importance, importance_path)