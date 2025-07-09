import os
import pandas as pd
import numpy as np
import datetime
import time
import logging
import psycopg
import joblib
import pytz
from dotenv import load_dotenv

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping


def parse_price(x):
    try:
        return float(str(x).replace('$', '').replace('USD', '').strip())
    except Exception:
        return np.nan


def transform_review_column(df, tfidf=None, svd=None):
    df['review'] = df['review'].fillna('')
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df['review'])
    else:
        tfidf_matrix = tfidf.transform(df['review'])

    if svd is None:
        svd = TruncatedSVD(n_components=1, random_state=42)
        review_score = svd.fit_transform(tfidf_matrix).flatten()
    else:
        review_score = svd.transform(tfidf_matrix).flatten()

    df['review_score'] = review_score
    return df, tfidf, svd


class CompetitorPricingTransformer:
    def __init__(self):
        self.tag_price_map = {}

    def fit(self, df):
        tag_prices = {}
        for tags, price in zip(df['tags'], df['current_price']):
            for tag in tags:
                tag_prices.setdefault(tag, []).append(price)
        self.tag_price_map = {tag: np.median(prices) for tag, prices in tag_prices.items()}
        return self

    def transform(self, df):
        def competitor_price_for_tags(tags):
            prices = [self.tag_price_map.get(tag, np.nan) for tag in tags]
            prices = [p for p in prices if not np.isnan(p)]
            return np.mean(prices) if prices else np.nan

        df['competitor_pricing'] = df['tags'].apply(competitor_price_for_tags)
        median_price = df['current_price'].median()
        df['competitor_pricing'] = df['competitor_pricing'].fillna(median_price)
        return df


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
SEND_TIMEOUT = 10

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')


def get_connection(dbname=None, autocommit=False):
    db = dbname or POSTGRES_DB
    conn_str = (
        f"host={POSTGRES_HOST} port={POSTGRES_PORT} user={POSTGRES_USER} "
        f"password={POSTGRES_PASSWORD} dbname={db}"
    )
    return psycopg.connect(conn_str, autocommit=autocommit)


# Load your trained model pipeline, with error handling
try:
    model = joblib.load('models/discount_model_pipeline_small.pkl')
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Paths to save/load transformers
TFIDF_PATH = 'models/tfidf.pkl'
SVD_PATH = 'models/svd.pkl'
COMPETITOR_TRANSFORMER_PATH = 'models/competitor_transformer.pkl'
MLB_GENRES_PATH = 'models/mlb_genres.pkl'
MLB_TAGS_PATH = 'models/mlb_tags.pkl'

RAW_FEATURES = [
    'total_reviews', 'positive_percent', 'current_price', 'discounted_price',
    'owners_log_mean', 'days_after_publish', 'review', 'genres', 'tags'
]

# Global variables for transformers
tfidf = None
svd = None
competitor_transformer = None
mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()


def preprocess(df: pd.DataFrame, fit_transformers=False) -> pd.DataFrame:
    global tfidf, svd, competitor_transformer, mlb_genres, mlb_tags

    df = df.dropna(subset=[
        'total_reviews', 'positive_percent', 'genres', 'tags',
        'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
    ])

    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)
    df = df.dropna(subset=['current_price', 'discounted_price'])

    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip()])
    df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';') if t.strip()])

    df = df[df['genres'].map(len) > 0]
    df = df[df['tags'].map(len) > 0]
    df['review'] = df.get('review', '').fillna('')

    if fit_transformers:
        df, tfidf, svd = transform_review_column(df)
        joblib.dump(tfidf, TFIDF_PATH)
        joblib.dump(svd, SVD_PATH)
    else:
        if tfidf is None or svd is None:
            tfidf = joblib.load(TFIDF_PATH)
            svd = joblib.load(SVD_PATH)
        df, _, _ = transform_review_column(df, tfidf=tfidf, svd=svd)

    global competitor_transformer
    if fit_transformers:
        competitor_transformer = CompetitorPricingTransformer()
        competitor_transformer.fit(df)
        joblib.dump(competitor_transformer, COMPETITOR_TRANSFORMER_PATH)
    else:
        if competitor_transformer is None:
            competitor_transformer = joblib.load(COMPETITOR_TRANSFORMER_PATH)
    df = competitor_transformer.transform(df)

    if fit_transformers:
        mlb_genres.fit(df['genres'])
        mlb_tags.fit(df['tags'])
        joblib.dump(mlb_genres, MLB_GENRES_PATH)
        joblib.dump(mlb_tags, MLB_TAGS_PATH)
    else:
        # Safer check if classes are loaded, fallback to load
        if not (hasattr(mlb_genres, 'classes_') and len(mlb_genres.classes_) > 0):
            mlb_genres = joblib.load(MLB_GENRES_PATH)
        if not (hasattr(mlb_tags, 'classes_') and len(mlb_tags.classes_) > 0):
            mlb_tags = joblib.load(MLB_TAGS_PATH)

    df['genres'] = df['genres'].apply(lambda lst: [g for g in lst if g in mlb_genres.classes_])
    df['tags'] = df['tags'].apply(lambda lst: [t for t in lst if t in mlb_tags.classes_])

    genres_encoded = pd.DataFrame(mlb_genres.transform(df['genres']), columns=mlb_genres.classes_, index=df.index)
    tags_encoded = pd.DataFrame(mlb_tags.transform(df['tags']), columns=mlb_tags.classes_, index=df.index)

    df = pd.concat([df, genres_encoded, tags_encoded], axis=1)
    df = df.drop(columns=['genres', 'tags', 'review'], errors='ignore')
    df = df.loc[:, ~df.columns.duplicated()]

    MODEL_FEATURES = [
        'total_reviews', 'positive_percent', 'current_price', 'discounted_price',
        'owners_log_mean', 'days_after_publish', 'review_score', 'competitor_pricing'
    ] + list(mlb_genres.classes_) + list(mlb_tags.classes_)

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df.reindex(columns=MODEL_FEATURES)

    return df


def prep_db():
    with get_connection(dbname='postgres', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
            exists = cur.fetchone()
            if not exists:
                cur.execute("CREATE DATABASE test;")
    with get_connection(dbname='test', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DROP TABLE IF EXISTS discount_model_metrics;
                CREATE TABLE discount_model_metrics (
                    timestamp TIMESTAMP,
                    prediction_drift FLOAT,
                    num_drifted_columns INTEGER,
                    share_missing_values FLOAT
                )
            """)


def get_batch(i: int):
    df_raw = pd.read_csv("data/combined4.csv").sort_values('days_after_publish')
    batch_raw = df_raw[df_raw['days_after_publish'] == i]
    if batch_raw.empty:
        return None

    batch_raw = batch_raw.dropna(subset=RAW_FEATURES)
    batch_processed = preprocess(batch_raw, fit_transformers=False)
    batch_processed = batch_processed.dropna(subset=batch_processed.columns)
    batch_processed['prediction'] = model.predict(batch_processed)
    return batch_processed


def run_drift(ref: pd.DataFrame, cur: pd.DataFrame):
    for df in (ref, cur):
        for col in ['genres', 'tags', 'review']:
            df.pop(col, None)
        df.drop_duplicates(axis=1, inplace=True)

    DRIFT_FEATURES = list(dict.fromkeys(ref.columns))
    if 'prediction' in DRIFT_FEATURES:
        DRIFT_FEATURES.remove('prediction')

    for df in (ref, cur):
        for col in DRIFT_FEATURES + ['prediction']:
            if col not in df.columns:
                df[col] = 0

    expected_cols = DRIFT_FEATURES + ['prediction']

    # Safe reindexing fix to avoid column mismatch errors
    ref = ref.reindex(columns=expected_cols, fill_value=0)
    cur = cur.reindex(columns=expected_cols, fill_value=0)

    rpt = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    rpt.run(reference_data=ref, current_data=cur, column_mapping=ColumnMapping(
        prediction='prediction',
        numerical_features=DRIFT_FEATURES,
        target=None
    ))

    return rpt.as_dict()


def insert_sql(metrics, timestamp, cursor):
    cursor.execute(
        "INSERT INTO discount_model_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values) "
        "VALUES (%s, %s, %s, %s)",
        (
            timestamp,
            metrics['metrics'][0]['result']['drift_score'],
            metrics['metrics'][1]['result']['number_of_drifted_columns'],
            metrics['metrics'][2]['result']['current']['share_of_missing_values']
        )
    )


if __name__ == "__main__":
    try:
        reference_raw = pd.read_csv("data/combined4.csv")
        reference_raw = reference_raw.dropna(subset=RAW_FEATURES)
        reference = preprocess(reference_raw, fit_transformers=True)
        reference = reference.dropna(subset=reference.columns)
        reference['prediction'] = model.predict(reference)
    except Exception as e:
        logging.error(f"Error loading or preprocessing reference data: {e}")
        raise

    prep_db()

    last_send = datetime.datetime.now(pytz.UTC) - datetime.timedelta(seconds=SEND_TIMEOUT)

    with get_connection(dbname='test', autocommit=True) as conn:
        for day in range(30):
            batch = get_batch(day)
            if batch is None:
                logging.warning(f"Day {day} no data")
                continue

            drift_report = run_drift(reference.copy(), batch.copy())

            with conn.cursor() as cur:
                insert_sql(drift_report, last_send, cur)
                conn.commit()

            logging.info(f"Logged drift for day {day}")

            now = datetime.datetime.now(pytz.UTC)
            elapsed = (now - last_send).total_seconds()
            if elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - elapsed)
            last_send += datetime.timedelta(seconds=SEND_TIMEOUT)
