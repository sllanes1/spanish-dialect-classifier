from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import sqlite3
from datetime import datetime

app = FastAPI()

# Allows React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence TEXT NOT NULL,
            predicted_dialect TEXT NOT NULL,
            mx_probability REAL NOT NULL,
            es_probability REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load trained model
with open("model.pkl", "rb") as f:
    vectorizer, clf = pickle.load(f)

# Request shape
class SentenceRequest(BaseModel):
    sentence: str

# Predict endpoint
@app.post("/predict")
def predict(request: SentenceRequest):
    sentence = request.sentence.strip()

    if len(sentence.split()) < 4:
        return {"error": "Please enter at least 4 words."}
    
    vec = vectorizer.transform([sentence.lower()])
    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec)[0]

    classes = list(clf.classes_)
    mx_prob = prob[classes.index("mx")]
    es_prob = prob[classes.index("es")]

    # Save to database
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (sentence, predicted_dialect, mx_probability, es_probability, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (sentence, pred, float(mx_prob), float(es_prob), datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return {
        "predicted_dialect": pred.upper(),
        "mx_probability": round(float(mx_prob), 3),
        "es_probability": round(float(es_prob), 3),
    }

# History endpoint
@app.get("/history")
def history():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sentence, predicted_dialect, mx_probability, es_probability, timestamp FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"sentence": r[0], "dialect": r[1], "mx": r[2], "es": r[3], "time": r[4]}
        for r in rows
    ]