from flask import Flask, jsonify, request
from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import base64, io
import threading
import time
from datetime import datetime
import requests
import json

app = Flask(__name__)

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "RetailML"
COLLECTION = "Reviews"

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return img

@app.route("/sentiment-analysis", methods=["POST"])
def sentiment_analysis():
    try:
        # Connect to MongoDB and Load Data
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION]
        df = pd.DataFrame(list(collection.find({}, {"_id": 0})))

        if "ReviewText" not in df.columns or "Rating" not in df.columns:
            return jsonify({"error": "Missing 'ReviewText' or 'Rating' in MongoDB collection."}), 400

        # Label Sentiment from Ratings (1–2 = Negative, 3 = Neutral, 4–5 = Positive)
        def label_sentiment(rating):
            if rating >= 4:
                return 2  # Positive
            elif rating == 3:
                return 1  # Neutral
            else:
                return 0  # Negative

        df["Sentiment"] = df["Rating"].apply(label_sentiment)

        X = df["ReviewText"]
        y = df["Sentiment"]

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Build Pipelines
        model_choice = request.json.get("model", 1)

        # Pipeline 1 – Logistic Regression
        pipe1 = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000))
        ])

        # Pipeline 2 – Voting (Logistic + RandomForest)
        pipe2 = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("voting", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000)),
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
                ],
                voting="soft"
            ))
        ])

        # Pipeline 3 – Voting (Logistic + RF + Gradient Boosting)
        pipe3 = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
            ("voting", VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000)),
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                    ("gb", GradientBoostingClassifier(n_estimators=150, random_state=42))
                ],
                voting="soft"
            ))
        ])

        # Select pipeline
        if model_choice == 1:
            model = pipe1
            model_name = "Logistic Regression"
        elif model_choice == 2:
            model = pipe2
            model_name = "Voting (LR + RF)"
        else:
            model = pipe3
            model_name = "Voting (LR + RF + GB)"

        # Train Model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Confusion Matrix Plot
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Neg", "Neu", "Pos"],
                    yticklabels=["Neg", "Neu", "Pos"])
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        cm_image = fig_to_base64()

        # Predict New Reviews (optional input)
        new_reviews = request.json.get("new_reviews", [])
        preds = None
        if new_reviews:
            preds = model.predict(new_reviews).tolist()

        # Return Response
        result = {
            "status": "success",
            "model_used": model_name,
            "accuracy": round(acc, 3),
            "f1_score": round(f1, 3),
            "confusion_matrix_plot": cm_image,
            "new_reviews_predictions": preds
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =============================
# Run Server
# =============================
def run_flask():
    app.run(port=5003 , debug=False , use_reloader=False)

if __name__ == "__main__":
    run_flask()


# url = "http://127.0.0.1:5002/sentiment-analysis"

# data = {
#     "model": 3,  # choose 1, 2, or 3
#     "new_reviews": [
#         "I love this product, very useful!",
#         "It was okay, nothing special.",
#         "Terrible experience, not recommended."
#     ]
# }

# response = requests.post(url, json=data)

# if response.status_code == 200:
#     res = response.json()
#     print("✅ Success")
#     print("Model Used:", res["model_used"])
#     print("Accuracy:", res["accuracy"])
#     print("F1 Score:", res["f1_score"])
#     print("Predictions:", res["new_reviews_predictions"])
# else:
#     print("❌ Error:", response.status_code)
#     print(response.text)
