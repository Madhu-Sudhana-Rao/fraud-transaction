import pickle
import os
import numpy as np
import streamlit as st

MODEL_PATH = "fraud.pkl"

# Load Model
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            loaded_data = pickle.load(file)
            
            if isinstance(loaded_data, (tuple, list)) and len(loaded_data) > 0:
                model = loaded_data[0]  # Extract only the trained model
            elif hasattr(loaded_data, "predict"):
                model = loaded_data
            else:
                st.error(f"❌ Invalid model format: {type(loaded_data)}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ Model file not found! Please upload 'fraud.pkl'.")

# Streamlit UI
st.title("Fraud Detection System")

# Input fields
transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=0.1)
transaction_type = st.selectbox("Transaction Type", ["Bill Payment", "Bank Transfer", "ATM Withdrawal", "POS Payment", "Online Purchase"])
device_used = st.selectbox("Device Used", ["Desktop", "Mobile", "Tablet", "Unknown Device"])
time_of_transaction = st.number_input("Time of Transaction (24-hour format)", min_value=0.0, max_value=23.99, step=0.01)
previous_fraudulent_transactions = st.number_input("Previous Fraudulent Transactions", min_value=0, step=1)
account_age = st.number_input("Account Age (in days)", min_value=0, step=1)
number_of_transactions_24h = st.number_input("Number of Transactions in Last 24H", min_value=0, step=1)
payment_method = st.selectbox("Payment Method", ["UPI", "Debit Card", "Net Banking", "Credit Card", "Invalid Method"])

# Mappings
transaction_type_mapping = {
    "Bill Payment": 0,
    "Bank Transfer": 1,
    "ATM Withdrawal": 2,
    "POS Payment": 3,
    "Online Purchase": 4
}

device_used_mapping = {
    "Desktop": 0,
    "Mobile": 1,
    "Tablet": 2,
    "Unknown Device": 3
}

payment_method_mapping = {
    "UPI": 0,
    "Debit Card": 1,
    "Net Banking": 2,
    "Credit Card": 3,
    "Invalid Method": 4
}

# Predict button
if st.button("Predict Transaction Status"):
    if model is None or not hasattr(model, "predict"):
        st.error("❌ Invalid or missing model. Please upload a correct 'fraud.pkl'.")
    else:
        try:
            features = [
                transaction_amount,
                transaction_type_mapping.get(transaction_type, -1),
                time_of_transaction,
                device_used_mapping.get(device_used, -1),
                previous_fraudulent_transactions,
                account_age,
                number_of_transactions_24h,
                payment_method_mapping.get(payment_method, -1)
            ]
            
            if -1 in features[1:]:
                st.error("❌ Invalid transaction type, device, or payment method.")
            else:
                input_data = np.array(features).reshape(1, -1)
                prediction = model.predict(input_data)[0]
                result = "⚠️ Fraudulent Transaction Detected" if prediction == 1 else "✅ Transaction is Legitimate"
                st.success(result)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
