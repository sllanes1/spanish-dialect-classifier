from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from datetime import datetime, timedelta
from jose import jwt
import pickle
import os
import sqlite3
import numpy as np
import anthropic
import bcrypt

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
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
    conn = sqlite3.connect("dialect.db")
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

    # users table 
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
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

# Signup request shape
class SignupRequest(BaseModel):
    username: str
    password: str

# Login request shape
class LoginRequest(BaseModel):
    username: str
    password: str

# Connect with Claude API
def get_claude_explanation(sentence, predicted_dialect, mx_prob, es_prob, top_mx_features, top_es_features):
    prompt = f"""
    A Spanish dialect classifier analyzed this sentence: "{sentence}"
    It predicted: {predicted_dialect} Spanish
    MX probability: {mx_prob}
    ES probability: {es_prob}
    
    Top patterns pushing toward MX: {top_mx_features}
    Top patterns pushing toward ES: {top_es_features}
    
    In 2-3 friendly sentences can you explain to a non-technical user why this sentence 
    was classified as {predicted_dialect} Spanish. Focus on the actual words and 
    patterns, not technical jargon.
    """
    try: 
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Explanation currently unavailable: {str(e)}"

# Predict endpoint
@app.post("/predict")
def predict(request: SentenceRequest):
    sentence = request.sentence.strip()

    if len(sentence.split()) < 4:
        return {"error": "Please enter at least 4 words."}
    
    vec = vectorizer.transform([sentence.lower()])
    pred = clf.predict(vec)[0]
    prob = clf.predict_proba(vec)[0]

    # Finds the correct index dynamically
    classes = list(clf.classes_)        #  ['es', 'mx']
    mx_prob = prob[classes.index("mx")] # finds mx at index 1
    es_prob = prob[classes.index("es")] # finds es at index 0

    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = clf.coef_[0]
    sentence_vec = vec.toarray()[0]
    present_features = np.where(sentence_vec > 0)[0]
    present_coefs = coef[present_features]
    present_names = feature_names[present_features]

    # Sort and grab top 5
    top_mx_features = present_names[np.argsort(present_coefs)[-5:]][::-1]
    top_es_features = present_names[np.argsort(present_coefs)[:5]]

    # Claude explanation
    explanation = get_claude_explanation(
    sentence, pred, 
    round(float(mx_prob), 3), 
    round(float(es_prob), 3),
    top_mx_features.tolist(),
    top_es_features.tolist()
    )

    # Save to database
    conn = sqlite3.connect("dialect.db")
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

        "top_mx_features" : top_mx_features.tolist(),
        "top_es_features" : top_es_features.tolist(),
        "explanation" : explanation

    }

# History endpoint
@app.get("/history")
def history():
    conn = sqlite3.connect("dialect.db")
    cursor = conn.cursor()
    cursor.execute("SELECT sentence, predicted_dialect, mx_probability, es_probability, timestamp FROM predictions ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    return [
        {"sentence": r[0], "dialect": r[1], "mx": r[2], "es": r[3], "time": r[4]}
        for r in rows
    ]

# Signup endpoint
@app.post("/signup")
def signup(request: SignupRequest):
    conn = sqlite3.connect("dialect.db")
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ?", (request.username,))
    existing_user = cursor.fetchone()

    if existing_user:
        conn.close() 
        raise HTTPException(status_code=400, detail="Username already exists") 

    # hash password
    password_hash = bcrypt.hashpw(request.password.encode("utf-8"), bcrypt.gensalt())

    # save username and password to database
    cursor.execute("""
        INSERT INTO users (username, password_hash, created_at)
        VALUES (?, ?, ?)
    """, (request.username, password_hash, datetime.now().isoformat()))
    conn.commit()
    conn.close()
    
    return {"message": "Account created successfully!"}

# Login endpoint
@app.post("/login")
def login(request: LoginRequest):
    conn = sqlite3.connect("dialect.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (request.username,))
    user = cursor.fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    # check is password matches the stored hash
    password_matches = bcrypt.checkpw(request.password.encode("utf-8"), user[2])

    if not password_matches:
        conn.close()
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = jwt.encode(
    {"user_id": user[0], "username": user[1]},
    os.getenv("SECRET_KEY"),
    algorithm="HS256"
    )

    conn.close()
    return {"token": token, "username": user[1]}