from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from utils.transformers_new import CompetitorPricingTransformer

app = FastAPI()

# === Global Model Variables ===
model = None
tfidf = None
svd = None
mlb_genres = None
mlb_tags = None
competitor_transformer = None


class GameFeatures(BaseModel):
    name: str
    reviews: str
    all_tags: list[str]
    genres: list[str]
    release_date: str
    discounted_price: float
    initial_price: float
    metascore: float
    recommendations: int
    average_playtime: float
    median_playtime: float
    developers: str
    publishers: str
    windows: bool
    mac: bool
    linux: bool
    early_access: bool
    single_player: bool
    multi_player: bool
    vr_support: bool
    game_dlc: int
    positive_ratio: float
    sentiment: str
    competitor_discount: float

    def to_df(self):
        return pd.DataFrame([self.model_dump()])


@app.on_event("startup")
def load_model():
    global model, tfidf, svd, mlb_genres, mlb_tags, competitor_transformer
    try:
        model = joblib.load("models/discount_model_pipeline.pkl")
        tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        svd = joblib.load("models/svd_transform.pkl")
        mlb_genres = joblib.load("models/mlb_genres.pkl")
        mlb_tags = joblib.load("models/mlb_tags.pkl")
        competitor_transformer = CompetitorPricingTransformer()
    except Exception as e:
        raise RuntimeError(f"Failed to load model or components: {e}")


@app.post("/predict")
def predict_discount(data: GameFeatures):
    try:
        X = data.to_df()
        X["competitor_discount"] = competitor_transformer.transform(X)
        prediction = model.predict(X)
        return {"predicted_discount_pct": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
