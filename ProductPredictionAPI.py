from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import base64, io
import threading
import time
import requests


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


@app.route("/product-purchase-prediction", methods=["POST"])
def product_purchase_prediction():
    try:
        # Load Data from MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

        if df.empty:
            return jsonify({"status": "error", "message": "No data found in MongoDB"}), 400

        print(" Columns:", df.columns.tolist())

        #  Dynamic Preprocessing
        # Expected columns
        possible_features = ["Age", "Quantity", "UnitPrice", "TotalAmount", "TotalPrice"]

        # Detect available features
        available_features = [col for col in possible_features if col in df.columns]

        if not available_features:
            return jsonify({"status": "error", "message": "No valid numeric features found"}), 400

        # Prefer 'TotalAmount' as target; fallback to 'TotalPrice'
        target = "TotalAmount" if "TotalAmount" in df.columns else "TotalPrice"

        if target not in df.columns:
            return jsonify({"status": "error", "message": f"Missing target column '{target}'"}), 400

        # Clean dataset
        df = df.dropna(subset=available_features + [target])

        X = df[available_features]
        y = df[target]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        #  Choose Model
        model_choice = request.json.get("model", "rf").lower()

        if model_choice == "lr":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression())
            ])
            model_name = "Linear Regression"

        elif model_choice == "logistic":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LogisticRegression(max_iter=1000))
            ])
            model_name = "Logistic Regression"

        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", RandomForestRegressor(n_estimators=150, random_state=42))
            ])
            model_name = "Random Forest Regressor"

        #Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        #  EDA (Before Prediction)
        plt.figure(figsize=(8, 5))
        sns.histplot(df[target], bins=30, kde=True, color='skyblue')
        plt.title(f"Target Distribution: {target}")
        plt.xlabel(target)
        plt.ylabel("Frequency")
        plot_before = fig_to_base64()

        # EDA (After Prediction)
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.title(f"{model_name} - Actual vs Predicted")
        plt.xlabel(f"Actual {target}")
        plt.ylabel(f"Predicted {target}")
        plot_after = fig_to_base64()

        #  Predict for New Data
        new_data = request.json.get("new_data", [])
        preds = None
        if new_data:
            new_df = pd.DataFrame(new_data)
            for col in available_features:
                if col not in new_df.columns:
                    new_df[col] = 0
            preds = model.predict(new_df[available_features]).tolist()

        # Response
        result = {
            "status": "success",
            "model_used": model_name,
            "features_used": available_features,
            "target_used": target,
            "mse": round(mse, 3),
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
    app.run(port=5007 , debug=False, use_reloader=False)

if __name__ == "__main__":
    run_flask()

# =============================
# 🔍 Example Request
# =============================
# url = "http://127.0.0.1:5003/product-purchase-prediction"

# data = {
#     "model": "rf",
#     "new_data": [
#         {"Age": 25, "Quantity": 3, "UnitPrice": 450, "TotalAmount": 5000},
#         {"Age": 40, "Quantity": 6, "UnitPrice": 800, "TotalAmount": 12000}
#     ]
# }

# response = requests.post(url, json=data)

# if response.status_code == 200:
#     res = response.json()
#     print("✅ Success")
#     print("Model Used:", res["model_used"])
#     print("Features Used:", res["features_used"])
#     print("Target:", res["target_used"])
#     print("MSE:", res["mse"])
#     print("R² Score:", res["r2_score"])
#     print("Predictions:", res["new_predictions"])
# else:
#     print("❌ Error:", response.status_code)
#     print(response.text)
