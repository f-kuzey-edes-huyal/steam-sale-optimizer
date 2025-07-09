import os
import time
import pandas as pd
import numpy as np
import joblib
import psycopg
import datetime
import pytz
import logging
from dotenv import load_dotenv

from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping

# load environment variables from .env file
load_dotenv()

# config constants
DATA_PATH = "data/combined4.csv"          # path to clean reference dataset
DRIFTED_DATA_PATH = "data/drifted_data.csv"  # path to drifted dataset

# custom transformer for competitor pricing feature
class CompetitorPricingTransformer:
    def __init__(self):
        self.tag_price_map = {}

    def fit(self, df):
        # calculate median price per tag based on input dataframe
        tag_prices = {}
        for tags, price in zip(df['tags'], df['current_price']):
            for tag in tags:
                tag_prices.setdefault(tag, []).append(price)
        self.tag_price_map = {tag: np.median(prices) for tag, prices in tag_prices.items()}
        return self

    def transform(self, df):
        # compute average competitor price for each row's tags
        def competitor_price_for_tags(tags):
            prices = [self.tag_price_map.get(tag, np.nan) for tag in tags]
            prices = [p for p in prices if not np.isnan(p)]
            return np.mean(prices) if prices else np.nan

        df['competitor_pricing'] = df['tags'].apply(competitor_price_for_tags)
        median_price = df['current_price'].median()
        # fill missing competitor prices with median current price
        df['competitor_pricing'] = df['competitor_pricing'].fillna(median_price)
        return df

# file paths for models and transformers
MODEL_PATH = 'models/discount_model_pipeline.pkl'
TFIDF_PATH = 'models/tfidf_vectorizer.pkl'
SVD_PATH = 'models/svd_transform.pkl'
MLB_GENRES_PATH = 'models/mlb_genres.pkl'
MLB_TAGS_PATH = 'models/mlb_tags.pkl'
COMPETITOR_TRANS_PATH = 'models/competitor_pricing_transformer.pkl'

# load saved models and transformers from disk
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)
svd = joblib.load(SVD_PATH)
mlb_genres = joblib.load(MLB_GENRES_PATH)
mlb_tags = joblib.load(MLB_TAGS_PATH)
competitor_transformer = joblib.load(COMPETITOR_TRANS_PATH)

# setup logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# get postgres connection parameters from environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'monitoring_db')

def get_connection():
    # create and return postgres connection
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
        autocommit=True
    )

def preprocess_monitoring_data(df):
    # remove rows missing important columns
    df = df.dropna(subset=['total_reviews', 'positive_percent', 'genres', 'tags',
                           'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'])

    # function to clean price fields by removing $ and USD
    def parse_price(x):
        try:
            return float(str(x).replace('$', '').replace('USD', '').strip())
        except:
            return np.nan

    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)
    # drop rows where price parsing failed
    df = df.dropna(subset=['current_price', 'discounted_price'])

    # calculate discount percentage
    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

    # split genres and tags from strings into lists
    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';') if t.strip()])

    # handle missing review column safely
    df['review'] = df.get('review', '')
    df['review'] = df['review'].fillna('')

    if len(df) == 0:
        return df

    # convert review texts to numeric scores using TF-IDF and SVD
    tfidf_matrix = tfidf.transform(df['review'])
    df['review_score'] = svd.transform(tfidf_matrix).flatten()

    # add competitor pricing feature
    df = competitor_transformer.transform(df)

    # filter genres and tags to known classes only
    df['genres'] = df['genres'].apply(lambda lst: [g for g in lst if g in mlb_genres.classes_])
    df['tags'] = df['tags'].apply(lambda lst: [t for t in lst if t in mlb_tags.classes_])

    # one-hot encode genres and tags
    genres_encoded = pd.DataFrame(mlb_genres.transform(df['genres']), columns=mlb_genres.classes_, index=df.index)
    tags_encoded = pd.DataFrame(mlb_tags.transform(df['tags']), columns=mlb_tags.classes_, index=df.index)

    # add one-hot columns, remove original text columns
    df = pd.concat([df, genres_encoded, tags_encoded], axis=1)
    df = df.drop(columns=['genres', 'tags', 'review'], errors='ignore')

    # list all features expected by the model
    features = ['total_reviews', 'positive_percent', 'current_price', 'discounted_price',
                'owners_log_mean', 'days_after_publish', 'review_score', 'competitor_pricing'] + \
               list(mlb_genres.classes_) + list(mlb_tags.classes_)

    # ensure all features exist in dataframe
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # keep only expected features, replace inf and nan with zeros
    df = df[features]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df

def load_reference_data():
    # load reference (clean) data for drift detection (days_after_publish <= 5)
    df = pd.read_csv(DATA_PATH)
    ref_df = df[df['days_after_publish'] <= 5].copy()

    if ref_df.empty:
        logging.warning("Reference data empty, fallback to first 100 rows")
        ref_df = df.head(100).copy()

    ref_df = preprocess_monitoring_data(ref_df)
    if not ref_df.empty:
        ref_df['prediction'] = model.predict(ref_df)
    return ref_df

def load_batch(day: int):
    # load batch of drifted data for given 5-day window
    df = pd.read_csv(DRIFTED_DATA_PATH)
    lower = day * 5
    upper = (day + 1) * 5
    batch = df[(df['days_after_publish'] > lower) & (df['days_after_publish'] <= upper)].copy()
    if batch.empty:
        return None
    batch = preprocess_monitoring_data(batch)
    if not batch.empty:
        batch['prediction'] = model.predict(batch)
    return batch

def run_evidently_report(ref_df, current_df):
    # run drift detection report comparing reference and current data
    if ref_df.empty or current_df.empty:
        return None

    drift_features = [col for col in ref_df.columns if col != 'prediction']

    # make sure all columns exist in both datasets
    for df in (ref_df, current_df):
        for col in drift_features + ['prediction']:
            if col not in df.columns:
                df[col] = 0

    ref_df = ref_df.reindex(columns=drift_features + ['prediction'])
    current_df = current_df.reindex(columns=drift_features + ['prediction'])

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(reference_data=ref_df, current_data=current_df, column_mapping=ColumnMapping(
        prediction='prediction',
        numerical_features=drift_features
    ))

    return report.as_dict()

def store_metrics(metrics: dict, timestamp: datetime.datetime, cursor):
    # insert drift metrics into postgres table
    cursor.execute(
        """
        INSERT INTO model_monitoring_metrics (
            timestamp,
            prediction_drift,
            num_drifted_columns,
            share_missing_values
        ) VALUES (%s, %s, %s, %s)
        """,
        (
            timestamp,
            metrics['metrics'][0]['result']['drift_score'],
            metrics['metrics'][1]['result']['number_of_drifted_columns'],
            metrics['metrics'][2]['result']['current']['share_of_missing_values']
        )
    )

def prepare_db():
    # create metrics table if it doesn't exist yet
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_monitoring_metrics (
                    timestamp TIMESTAMP PRIMARY KEY,
                    prediction_drift FLOAT,
                    num_drifted_columns INTEGER,
                    share_missing_values FLOAT
                )
            """)

if __name__ == "__main__":
    logging.info("Starting model monitoring...")
    prepare_db()
    reference_data = load_reference_data()

    SEND_INTERVAL = 0  # change to 60 for production monitoring loop

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                day = 0
                while True:
                    batch = load_batch(day)
                    if batch is None:
                        logging.info(f"No data for day {day}, stopping monitoring.")
                        break

                    metrics = run_evidently_report(reference_data, batch)
                    if metrics is None:
                        logging.warning(f"Skipping day {day} because data empty.")
                        day += 1
                        continue

                    timestamp = datetime.datetime.now(pytz.UTC)
                    store_metrics(metrics, timestamp, cur)

                    drift_score = metrics['metrics'][0]['result']['drift_score']
                    num_drifted = metrics['metrics'][1]['result']['number_of_drifted_columns']
                    missing_pct = metrics['metrics'][2]['result']['current']['share_of_missing_values']

                    logging.info(f"Day {day} | Drift: {drift_score:.3f} | Drifted cols: {num_drifted} | Missing: {missing_pct:.2%}")

                    day += 1
                    time.sleep(SEND_INTERVAL)

    except Exception as e:
        logging.error(f"Monitoring failed: {e}")
        raise
