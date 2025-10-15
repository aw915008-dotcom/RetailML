from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
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
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return img_b64

@app.route("/demand-forecasting", methods=["POST"])
def demand_forecasting():
    try:
        #  Load Data
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

        if df.empty:
            return jsonify({"error": "No data found in MongoDB collection"}), 400

        print(" Columns From Mongo:", df.columns.tolist())

        # =============================
        # Preprocessing
        # =============================
        # If TotalAmount exists, use it for Demand proxy
        if "TotalAmount" not in df.columns and "TotalPrice" in df.columns:
            df["TotalAmount"] = df["TotalPrice"]

        # Create Demand column (simulate quantity ordered)
        if "Quantity" in df.columns:
            df["Demand"] = df["Quantity"]
        else:
            return jsonify({"error": "Missing 'Quantity' column for demand prediction"}), 400

        # Group by Product to aggregate demand info
        group_cols = [c for c in ["ProductID", "ProductName", "Category", "Brand"] if c in df.columns]
        if not group_cols:
            return jsonify({"error": "Missing product columns in MongoDB"}), 400

        product_df = df.groupby(group_cols).agg({
            "Quantity": "sum",
            "TotalAmount": "sum",
            "UnitPrice": "mean",
            "Demand": "sum"
        }).reset_index()

        print(" Product-level Columns:", product_df.columns.tolist())

        # =============================
        # Features & Target
        # =============================
        X = product_df[["TotalAmount", "UnitPrice"]]
        y = product_df["Demand"]

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # =============================
        #  Build & Train Model
        # =============================
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # =============================
        # Evaluation
        # =============================
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # =============================
        #  EDA Before Prediction
        # =============================
        plt.figure(figsize=(7,5))
        sns.histplot(product_df["Demand"], bins=20, kde=True, color="lightgreen")
        plt.title("Demand Distribution (Before Prediction)")
        plt.xlabel("Demand")
        plt.ylabel("Frequency")
        plot_before = fig_to_base64()

        # =============================
        # EDA After Prediction
        # =============================
        plt.figure(figsize=(7,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color="blue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title("Actual vs Predicted Demand")
        plt.xlabel("Actual Demand")
        plt.ylabel("Predicted Demand")
        plot_after = fig_to_base64()

        # =============================
        # New Predictions
        # =============================
        new_data = request.json.get("new_data", [])
        preds = None
        if new_data:
            new_df = pd.DataFrame(new_data)
            # Ensure all columns exist
            for col in ["TotalAmount", "UnitPrice"]:
                if col not in new_df.columns:
                    new_df[col] = 0
            preds = pipeline.predict(new_df[["TotalAmount", "UnitPrice"]]).tolist()

        # =============================
        #  Response
        # =============================
        result = {
            "status": "success",
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2_score": round(r2, 3),
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
    app.run(port=5006  , debug=False , use_reloader=False)

if __name__ == "__main__":
    run_flask()


# url = "http://127.0.0.1:5005/demand-forecasting"

# data = {
#     "new_data": [
#         {"TotalAmount": 30000, "UnitPrice": 150.0},
#         {"TotalAmount": 70000, "UnitPrice": 200.0}
#     ]
# }

# response = requests.post(url, json=data)

# if response.status_code == 200:
#     res = response.json()
#     print("✅ Success")
#     print("MAE:", res["mae"])
#     print("RMSE:", res["rmse"])
#     print("R²:", res["r2_score"])
#     print("Predicted Demands:", res["new_predictions"])
# else:
#     print("❌ Error:", response.status_code)
#     print(response.text)
