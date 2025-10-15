from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

@app.route("/customer-segmentation-classification", methods=["POST"])
def customer_segmentation_classification():
    try:
        # Load Data
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

        features = ["Age", "TotalAmount", "Quantity", "UnitPrice"]
        target = "ClusterLabel"

        # Check if Cluster Labels exist
        if target not in df.columns:
            from sklearn.cluster import KMeans
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[features])
            df[target] = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)

        df = df.dropna(subset=features + [target])
        X = df[features]
        y = df[target]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Choose Model (based on request or default)
        model_choice = request.json.get("model", "rf").lower()  # 'rf' or 'logistic'

        if model_choice == "logistic":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ])
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))
            ])

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)

        # Confusion Matrix Plot
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{'Random Forest' if model_choice=='rf' else 'Logistic Regression'} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_image = fig_to_base64()

        # Predict New Customers (optional input)
        new_customers = request.json.get("new_customers", [])
        preds = None
        if new_customers:
            new_df = pd.DataFrame(new_customers)
            preds = model.predict(new_df[features]).tolist()

        # Response
        result = {
            "status": "success",
            "model_used": "Random Forest" if model_choice == "rf" else "Logistic Regression",
            "accuracy": round(acc, 3),
            "f1_score": round(f1, 3),
            "classification_report": report,
            "confusion_matrix_plot": cm_image,
            "new_customer_predictions": preds
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================
#  Run Server
# =============================
def run_flask():
    app.run(port=5002 , debug=False , use_reloader=False)

if __name__ == "__main__":
    run_flask()

# # =============================
# # API URL
# # =============================
# url = "http://127.0.0.1:5001/customer-segmentation-classification"

# # =============================
# # Request Data
# # =============================
# data = {
#     "model": "rf",   # اختياري: 'rf' أو 'logistic'
#     "new_customers": [
#         {"Age": 25, "TotalAmount": 1500, "Quantity": 3, "UnitPrice": 400},
#         {"Age": 45, "TotalAmount": 9000, "Quantity": 6, "UnitPrice": 800}
#     ]
# }

# # =============================
# # Send Request
# # =============================
# response = requests.post(url, json=data)

# # =============================
# # Show Result
# # =============================
# if response.status_code == 200:
#     result = response.json()
#     print("✅ Request Successful")
#     print("Model Used:", result["model_used"])
#     print("Accuracy:", result["accuracy"])
#     print("F1 Score:", result["f1_score"])
#     print("Predicted Clusters:", result["new_customer_predictions"])
# else:
#     print("❌ Error:", response.status_code)
#     print(response.text)
