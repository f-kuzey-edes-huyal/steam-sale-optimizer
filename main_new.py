from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.transformers import CompetitorPricingTransformer
from config.preprocessing import NUMERIC_FEATURES  # Make sure this import works and contains correct features

app = FastAPI()

# Global variables for models and transformers
model = None
tfidf = None
svd = None
mlb_genres = None
mlb_tags = None
competitor_transformer = None

@app.on_event("startup")
def load_initial_models():
    reload_all_models()

def reload_all_models():
    global model, tfidf, svd, mlb_genres, mlb_tags, competitor_transformer
    model = joblib.load("models/discount_model_pipeline_local.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer_local.pkl")
    svd = joblib.load("models/svd_transform_local.pkl")
    mlb_genres = joblib.load("models/mlb_genres_local.pkl")
    mlb_tags = joblib.load("models/mlb_tags_local.pkl")
    competitor_transformer = joblib.load("models/competitor_pricing_transformer_local.pkl")

@app.post("/reload_model")
def reload_model():
    try:
        reload_all_models()
        return {"message": "Model reloaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

class GameData(BaseModel):
    game_id: int
    name: str
    release_date: str
    total_reviews: int
    positive_percent: int
    genres: str
    tags: str
    current_price: str
    discounted_price: str
    owners: str
    days_after_publish: int
    review: str
    owner_min: float
    owner_max: float
    owners_log_mean: float

def parse_price(val):
    try:
        return float(str(val).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan

def preprocess_input(data: GameData) -> pd.DataFrame:
    df = pd.DataFrame([data.dict()])

    # Parse prices
    df["current_price"] = df["current_price"].apply(parse_price)
    df["discounted_price"] = df["discounted_price"].apply(parse_price)
    df["discount_pct"] = 1 - (df["discounted_price"] / df["current_price"])

    # Process review text
    tfidf_matrix = tfidf.transform(df["review"].fillna(""))
    df["review_score"] = svd.transform(tfidf_matrix).flatten()

    # Process genres and tags as lists
    df["genres"] = df["genres"].fillna("").apply(lambda x: [g.strip() for g in x.split(",")])
    df["tags"] = df["tags"].fillna("").apply(lambda x: [t.strip() for t in x.split(";")])

    # One-hot encode genres and tags
    genres_encoded = pd.DataFrame(mlb_genres.transform(df["genres"]), columns=mlb_genres.classes_)
    tags_encoded = pd.DataFrame(mlb_tags.transform(df["tags"]), columns=mlb_tags.classes_)

    # Apply competitor pricing transformer
    df = competitor_transformer.transform(df)

    # Concatenate encoded columns
    df = pd.concat([df.reset_index(drop=True), genres_encoded, tags_encoded], axis=1)

    # Select features consistent with training
    all_features = NUMERIC_FEATURES + ['review_score', 'competitor_pricing'] + list(genres_encoded.columns) + list(tags_encoded.columns)

    # Check if all_features exist in df.columns to avoid KeyError
    missing_features = set(all_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing expected features: {missing_features}")

    return df[all_features]

@app.post("/predict")
def predict(data: GameData):
    try:
        df_prepared = preprocess_input(data)
        prediction = model.predict(df_prepared)
        return {"predicted_discount_pct": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
