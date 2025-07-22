from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from config.preprocessing import NUMERIC_FEATURES

app = FastAPI()

# === Global Model Variables ===
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
    model = joblib.load("models/discount_model_pipeline.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    svd = joblib.load("models/svd_transform.pkl")
    mlb_genres = joblib.load("models/mlb_genres.pkl")
    mlb_tags = joblib.load("models/mlb_tags.pkl")
    competitor_transformer = joblib.load(
        "models/competitor_pricing_transformer.pkl"
    )
    
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
    except Exception:
        return None


def preprocess_input(data: GameData) -> pd.DataFrame:
    df = pd.DataFrame([data.model_dump()])
    df["current_price"] = df["current_price"].apply(parse_price)
    df["discounted_price"] = df["discounted_price"].apply(parse_price)
    df["discount_pct"] = 1 - (df["discounted_price"] / df["current_price"])

    tfidf_matrix = tfidf.transform(df["review"].fillna(""))
    df["review_score"] = svd.transform(tfidf_matrix).flatten()

    df["genres"] = df["genres"].fillna("").apply(
        lambda x: [g.strip() for g in x.split(",")]
    )
    df["tags"] = df["tags"].fillna("").apply(
        lambda x: [t.strip() for t in x.split(";")]
    )

    genres_encoded = pd.DataFrame(
        mlb_genres.transform(df["genres"]), columns=mlb_genres.classes_
    )
    tags_encoded = pd.DataFrame(
        mlb_tags.transform(df["tags"]), columns=mlb_tags.classes_
    )

    competitor_df = competitor_transformer.transform(df)

    df = pd.concat(
        [
            competitor_df.reset_index(drop=True),
            genres_encoded,
            tags_encoded
            ],
        axis=1
    )

    all_features = (
        NUMERIC_FEATURES
        + ["review_score", "competitor_pricing"]
        + list(mlb_genres.classes_)
        + list(mlb_tags.classes_)
    )

    return df[all_features]


@app.post("/predict")
def predict(data: GameData):
    try:
        df_prepared = preprocess_input(data)
        prediction = model.predict(df_prepared)
        return {"predicted_discount_pct": float(prediction[0])}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}
