import joblib

# Load saved model
model = joblib.load("churn_model.pkl")

print("Saved model loaded successfully!")
# Sample customer data

new_customer = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 427.50
}

print("\nSample customer data created!")
import pandas as pd

# Convert customer data into DataFrame
customer_df = pd.DataFrame([new_customer])

# Predict customer churn
prediction = model.predict(customer_df)

# Predict probability
probability = model.predict_proba(customer_df)

print("\nCustomer Churn Prediction:")

if prediction[0] == "Yes":
    print("Customer may leave the company!")
else:
    print("Customer may stay with the company!")

print("\nChurn Probability:", round(probability[0][1] * 100, 2), "%")