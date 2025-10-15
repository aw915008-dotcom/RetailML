from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64
import threading
import time
from datetime import datetime
import requests
import json
import base64


app = Flask(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "RetailML"
COLLECTION = "Transactions"


def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return encoded

@app.route("/customer-segmentation", methods=["POST"])
def customer_segmentation():
    try:
        #  Load data from MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

        #  Select features
        features = ["Age", "TotalAmount", "Quantity", "UnitPrice"]
        df = df.dropna(subset=features)
        X = df[features]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find best K (2–9)
        best_k, best_score = 0, -1
        for k in range(2, 6):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score, best_k = score, k

        #  Train final KMeans
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        df["ClusterLabel"] = kmeans.fit_predict(X_scaled)

        # Cluster summary
        summary = df.groupby("ClusterLabel")[features].mean().reset_index().to_dict(orient="records")

        # EDA Plots (Base64)
        # Scatter Plot: Age vs TotalAmount
        plt.figure(figsize=(6,5))
        sns.scatterplot(x="Age", y="TotalAmount", hue="ClusterLabel", data=df, palette="Set2")
        plt.title("Customer Segmentation (Age vs TotalAmount)")
        scatter_plot_1 = plot_to_base64()

        # Scatter Plot: Quantity vs TotalAmount
        plt.figure(figsize=(6,5))
        sns.scatterplot(x="Quantity", y="TotalAmount", hue="ClusterLabel", data=df, palette="Set1")
        plt.title("Customer Segmentation (Quantity vs TotalAmount)")
        scatter_plot_2 = plot_to_base64()

        #  Return JSON response
        response = {
            "status": "success",
            "optimal_k": best_k,
            "silhouette_score": round(best_score, 3),
            "cluster_summary": summary,
            "sample_data": df.head(10).to_dict(orient="records"),
            "plots": {
                "Age_vs_TotalAmount": scatter_plot_1,
                "Quantity_vs_TotalAmount": scatter_plot_2
            }
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================
# Run Server
# =============================
def run_flask():
    app.run(port=5001 , debug=False , use_reloader=False)

if __name__ == "__main__":
    run_flask()



# =============================
# 1. Define API Endpoint
# =============================
# url = "http://127.0.0.1:5000/customer-segmentation"

# # =============================
# # 2. Send POST Request
# # =============================
# response = requests.post(url)

# # =============================
# # 3. Parse Response
# # =============================
# if response.status_code == 200:
#     data = response.json()
    
#     print("✅ Request Successful")
#     print("Optimal K:", data["optimal_k"])
#     print("Silhouette Score:", data["silhouette_score"])
#     print("\nCluster Summary:")
#     for cluster in data["cluster_summary"]:
#         print(cluster)

#     # Example: Decode and save one plot
#     plot_data = data["plots"]["Age_vs_TotalAmount"]
#     img_bytes = base64.b64decode(plot_data)
#     with open("Age_vs_TotalAmount.png", "wb") as f:
#         f.write(img_bytes)
#     print("\n📊 Saved plot image: Age_vs_TotalAmount.png")

# else:
#     print("❌ Request Failed")
#     print("Status Code:", response.status_code)
#     print("Message:", response.text)
