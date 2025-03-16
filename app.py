import pickle
import os
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = "fraud.pkl"

# Load Model
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            loaded_data = pickle.load(file)

            # Ensure loaded model is not a string or incorrect format
            if isinstance(loaded_data, (tuple, list)) and len(loaded_data) > 0:
                model = loaded_data[0]  # Extract only the trained model
            elif hasattr(loaded_data, "predict"):
                model = loaded_data  # If it has 'predict', it's a valid model
            else:
                print(f"❌ Invalid model format: {type(loaded_data)}")
                model = None

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ Model file not found! Please upload 'fraud.pkl'.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or not hasattr(model, "predict"):
        return render_template("output.html", prediction="❌ Invalid or missing model. Please upload a correct 'fraud.pkl'.")

    try:
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

        # Extract and validate form inputs
        try:
            features = [
                float(request.form.get("Transaction_Amount", 0)),  # Default to 0 if missing
                transaction_type_mapping.get(request.form.get("Transaction_Type"), -1),
                float(request.form.get("Time_of_Transaction", 0)),  # Default to 0
                device_used_mapping.get(request.form.get("Device_Used"), -1),
                int(request.form.get("Previous_Fraudulent_Transactions", 0)),  # Default to 0
                int(request.form.get("Account_Age", 0)),  # Default to 0
                int(request.form.get("Number_of_Transactions_Last_24H", 0)),  # Default to 0
                payment_method_mapping.get(request.form.get("Payment_Method"), -1)
            ]
        except ValueError:
            return render_template("output.html", prediction="❌ Invalid input detected. Please enter valid values.")

        # Check for invalid categorical values (-1 means not found in mapping)
        if -1 in features[1:]:
            return render_template("output.html", prediction="❌ Invalid transaction type, device, or payment method.")

        # Convert to NumPy array for prediction
        input_data = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]
        result = "⚠️ Fraudulent Transaction Detected" if prediction == 1 else "✅ Transaction is Legitimate"

    except Exception as e:
        result = f"❌ Error: {str(e)}"

    return render_template("output.html", prediction=result)

if __name__ == "__main__":
    if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

