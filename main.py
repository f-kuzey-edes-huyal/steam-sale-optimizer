from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load saved components
model = joblib.load("models/discount_model_pipeline.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
svd = joblib.load("models/svd_transform.pkl")
mlb_genres = joblib.load("models/mlb_genres.pkl")
mlb_tags = joblib.load("models/mlb_tags.pkl")
competitor_transformer = joblib.load("models/competitor_pricing_transformer.pkl")

app = FastAPI(title="Steam Discount Predictor API")

class PredictionRequest(BaseModel):
    total_reviews: int
    positive_percent: float
    current_price: float
    discounted_price: float
    owners_log_mean: float
    days_after_publish: int
    genres: list[str]
    tags: list[str]
    review: str

@app.get("/")
def root():
    return {"message": "Steam Discount Prediction API is live!"}

@app.post("/predict")
def predict_discount(data: PredictionRequest):
    try:
        df = pd.DataFrame([data.dict()])

        # Manual feature engineering to match training pipeline
        df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])
        df['genres'] = [data.genres]
        df['tags'] = [data.tags]

        tfidf_matrix = tfidf.transform(df['review'])
        df['review_score'] = svd.transform(tfidf_matrix).flatten()

        df = competitor_transformer.transform(df)

        # One-hot encode genres and tags using previously fit encoders
        genres_encoded = pd.DataFrame(mlb_genres.transform(df['genres']), columns=mlb_genres.classes_)
        tags_encoded = pd.DataFrame(mlb_tags.transform(df['tags']), columns=mlb_tags.classes_)

        df = df.reset_index(drop=True)
        df_final = pd.concat([df, genres_encoded, tags_encoded], axis=1)

        # Define required feature columns from training
        feature_cols = model.named_steps['preprocess'].get_feature_names_out()

        # Ensure all required columns are in df_final (missing ones get 0)
        for col in feature_cols:
            if col not in df_final.columns:
                df_final[col] = 0
        df_final = df_final[feature_cols]

        # Predict
        pred = model.predict(df_final)[0]
        return {"predicted_discount": round(pred, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
