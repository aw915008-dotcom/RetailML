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


app = Flask(__name__)

# =============================
# Registered APIs
# =============================
baseUrl = "http://127.0.0.1"
SERVICES = {
    "customer_segmentation_labeling": f"{baseUrl}:5001/customer-segmentation-labeling",
    "customer_segmentation_classification": f"{baseUrl}:5002/customer-segmentation-classification",
    "sentiment_analysis": f"{baseUrl}:5003/sentiment-analysis",
    "product_purchase_prediction": f"{baseUrl}:5007/product-purchase-prediction",
    "branch_profit_prediction": f"{baseUrl}:5005/branch-profit-prediction",
    "demand_forecasting": f"{baseUrl}:5006/demand-forecasting"
}


@app.route("/gateway", methods=["POST"])
def gateway():
    try:
        data = request.json
        if not data or "service" not in data:
            return jsonify({"error": "Missing 'service' field in request"}), 400

        service_name = data["service"]
        service_url = SERVICES.get(service_name)

        if not service_url:
            return jsonify({"error": f"Service '{service_name}' not found"}), 404

        # Remove 'service' key before forwarding
        payload = {k: v for k, v in data.items() if k != "service"}

        print(f"Forwarding request to: {service_name} → {service_url}")

        # Forward POST request
        response = requests.post(service_url, json=payload)

        try:
            res_json = response.json()
        except Exception:
            res_json = {"message": response.text}

        return jsonify({
            "gateway_status": "success",
            "service_called": service_name,
            "status_code": response.status_code,
            "response": res_json
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# =============================
# Run Gateway Server
# =============================
def run_flask():
    print("🚀 API Gateway running on port 5050...")
    app.run(port=5050  , debug=False , use_reloader=False)

if __name__ == "__main__":
    run_flask()
