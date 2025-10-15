from flask import Flask, request, jsonify
import requests
import threading
import time
from datetime import datetime
import requests
import json
import base64
from io import BytesIO
from PIL import Image



url = "http://127.0.0.1:5050/gateway"
# =============================
#  Choose which API to call
# =============================
service_name = input("""
Select API to call via Gateway:
1️⃣ customer_segmentation_labeling
2️⃣ customer_segmentation_classification
3️⃣ sentiment_analysis
4️⃣ product_purchase_prediction
5️⃣ branch_profit_prediction
6️⃣ demand_forecasting
👉 Enter number: """)
services = {
    "1": "customer_segmentation_labeling",
    "2": "customer_segmentation_classification",
    "3": "sentiment_analysis",
    "4": "product_purchase_prediction",
    "5": "branch_profit_prediction",
    "6": "demand_forecasting"
}
service = services.get(service_name)
if not service:
    print(" Invalid selection. Please run again.")
    exit()
print(f"\nCalling Service: {service}")
print("=" * 60)
# =============================
#  Example data (adjust automatically)
# =============================
# You can easily adjust these if API expects other inputs
data = {
    "service": service
}
# Example for predictive APIs
if "prediction" in service or "forecasting" in service:
    data["new_data"] = [
        {"TotalAmount": 30000, "UnitPrice": 150.0},
        {"TotalAmount": 70000, "UnitPrice": 200.0}
    ]
# Example for sentiment analysis
elif "sentiment" in service:
    data["new_reviews"] = [
        "The product was amazing!",
        "Worst purchase ever.",
        "It was okay, not great."
    ]
# Example for segmentation
elif "segmentation" in service:
    data["params"] = {"clusters": 3}
# =============================
#  Send Request
# =============================
response = requests.post(url, json=data)
# =============================
# Display Results + Decode Images
# =============================
if response.status_code == 200:
    res = response.json()
    print("\n GATEWAY REQUEST SUCCESS")
    print("=" * 60)
    print(f"📡 Service Called: {res.get('service_called', 'N/A')}")
    print("-" * 60)
    api_res = res.get("response", {})
    #  Print all non-image fields neatly
    for key, val in api_res.items():
        if isinstance(val, (float, int, str, list)) and not str(key).startswith("plot_"):
            print(f"{key:>25}: {val}")
    print("-" * 60)
    #  Decode & Show Images if present
    for key in api_res.keys():
        if "plot" in key and isinstance(api_res[key], str):
            try:
                img_data = base64.b64decode(api_res[key])
                img = Image.open(BytesIO(img_data))
                print(f" Displaying: {key.replace('_', ' ').title()}")
                img.show()
            except Exception as e:
                print(f" Could not display {key}: {e}")
    print("=" * 60)
else:
    print("❌ GATEWAY REQUEST FAILED")
    print("=" * 60)
    print(f"Status Code: {response.status_code}")
    print("Response Text:\n", response.text)
    print("=" * 60)