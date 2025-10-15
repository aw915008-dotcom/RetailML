from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64, io
import threading
import time
from datetime import datetime
import requests
import json


app = Flask(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "RetailML"
COLLECTION = "Transactions"

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return img


@app.route("/branch-profit-prediction", methods=["POST"])
def branch_profit_prediction():
    try:
        # Load Data
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

        if df.empty:
            return jsonify({"error": "No data found in MongoDB collection."}), 400

        print(" Raw Columns From Mongo:", df.columns.tolist())

        # =============================
        # Preprocessing (Supports TotalAmount & TotalPrice)
        # =============================
        if "Profit" not in df.columns:
            if "TotalAmount" in df.columns:
                df["Profit"] = df["TotalAmount"] * 0.2
            elif "TotalPrice" in df.columns:
                df["Profit"] = df["TotalPrice"] * 0.2
            else:
                return jsonify({"error": "No TotalAmount or TotalPrice found to calculate Profit"}), 400

        # Aggregate safely
        agg_dict = {}
        if "TotalAmount" in df.columns: agg_dict["TotalAmount"] = "sum"
        if "TotalPrice" in df.columns: agg_dict["TotalPrice"] = "sum"
        if "Quantity" in df.columns: agg_dict["Quantity"] = "sum"
        if "UnitPrice" in df.columns: agg_dict["UnitPrice"] = "mean"
        agg_dict["Profit"] = "sum"

        # Verify branch columns
        branch_keys = [c for c in ["BranchID", "BranchName", "BranchCity"] if c in df.columns]
        if not branch_keys:
            return jsonify({"error": "Missing branch identification columns (BranchID, BranchName, BranchCity)."}), 400

        branch_df = df.groupby(branch_keys).agg(agg_dict).reset_index()

        print("Branch-level Columns:", branch_df.columns.tolist())

        # Detect numeric features
        possible_features = [col for col in branch_df.columns if col.lower() in 
                             ["totalamount", "totalprice", "quantity", "unitprice"]]
        if not possible_features:
            return jsonify({"error": "No valid numeric features found for training"}), 400

        print("Detected Features:", possible_features)

        X = branch_df[possible_features]
        y = branch_df["Profit"]

        # =============================
        # Split Data
        # =============================
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =============================
        # Choose Model
        # =============================
        model_choice = request.json.get("model", "gb").lower()

        if model_choice == "linear":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression())
            ])
            model_name = "Linear Regression"

        elif model_choice == "rf":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(n_estimators=150, random_state=42))
            ])
            model_name = "Random Forest Regressor"

        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
            ])
            model_name = "Gradient Boosting Regressor"

        # =============================
        # Train Model
        # =============================
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # =============================
        # EDA Plots
        # =============================
        plt.figure(figsize=(7,5))
        sns.histplot(branch_df["Profit"], bins=20, kde=True, color="skyblue")
        plt.title("Branch Profit Distribution (Before Prediction)")
        plt.xlabel("Profit")
        plt.ylabel("Frequency")
        plot_before = fig_to_base64()

        plt.figure(figsize=(7,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="teal")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(f"{model_name} - Actual vs Predicted Profit")
        plt.xlabel("Actual Profit")
        plt.ylabel("Predicted Profit")
        plot_after = fig_to_base64()

        # =============================
        # Predict for New Branches
        # =============================
        new_data = request.json.get("new_data", [])
        preds = None
        if new_data:
            new_df = pd.DataFrame(new_data)

            # Convert TotalAmount → TotalPrice if needed
            if "TotalAmount" in new_df.columns and "TotalPrice" not in new_df.columns:
                new_df["TotalPrice"] = new_df["TotalAmount"]

            # Ensure all columns exist
            for col in possible_features:
                if col not in new_df.columns:
                    new_df[col] = 0

            preds = model.predict(new_df[possible_features]).tolist()

        # =============================
        # Response
        # =============================
        result = {
            "status": "success",
            "model_used": model_name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 3),
            "detected_features": possible_features,
            "plot_before_prediction": plot_before,
            "plot_after_prediction": plot_after,
            "new_predictions": preds
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =============================
# Run Server
# =============================
def run_flask():
    app.run(port=5005  , debug=False , use_reloader=False)

if __name__ == "__main__":
    run_flask()

# url = "http://127.0.0.1:5004/branch-profit-prediction"

# data = {
#     "model": "gb",  # or "rf" or "linear"
#     "new_data": [
#         {"TotalAmount": 50000, "TotalPrice": 52000, "Quantity": 120, "UnitPrice": 250.0},
#         {"TotalAmount": 150000, "TotalPrice": 155000, "Quantity": 400, "UnitPrice": 180.0}
#     ]
# }

# response = requests.post(url, json=data)

# if response.status_code == 200:
#     res = response.json()
#     print("✅ Success")
#     print("Detected Features:", res["detected_features"])
#     print("Model Used:", res["model_used"])
#     print("MAE:", res["mae"])
#     print("RMSE:", res["rmse"])
#     print("R²:", res["r2_score"])
#     print("Predictions:", res["new_predictions"])
# else:
#     print("❌ Error:", response.status_code)
#     print(response.text)
