import os
import pandas as pd
import numpy as np
import datetime
import time
import logging
import psycopg  # Make sure this is installed and imported!
import joblib
import pytz
from dotenv import load_dotenv  # To load env variables from .env

from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping

# Load environment variables from .env
load_dotenv()

# Read DB credentials from environment variables
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'postgres')

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
SEND_TIMEOUT = 10  # seconds

# Load model and transformers
model = joblib.load('models/discount_model_pipeline.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
svd = joblib.load('models/svd_transform.pkl')
mlb_genres = joblib.load('models/mlb_genres.pkl')
mlb_tags = joblib.load('models/mlb_tags.pkl')
competitor_transformer = joblib.load('models/competitor_pricing_transformer.pkl')

# Features expected after preprocessing (numeric + engineered)
num_features = ['total_reviews', 'positive_percent', 'current_price', 'discounted_price',
                'owners_log_mean', 'days_after_publish', 'review_score', 'competitor_pricing']

genre_cols = list(mlb_genres.classes_)
tag_cols = list(mlb_tags.classes_)

all_features = num_features + genre_cols + tag_cols

# Column mapping for Evidently
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=all_features,
    target=None
)

# SQL for metrics table
CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS discount_model_metrics;
CREATE TABLE discount_model_metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT
)
"""

def get_connection(dbname=None, autocommit=False):
    db = dbname if dbname else POSTGRES_DB
    conn_string = (
        f"host={POSTGRES_HOST} port={POSTGRES_PORT} user={POSTGRES_USER} "
        f"password={POSTGRES_PASSWORD} dbname={db}"
    )
    return psycopg.connect(conn_string, autocommit=autocommit)

def parse_price(val):
    try:
        return float(str(val).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan

def preprocess_data(df):
    df = df.dropna(subset=[
        'total_reviews', 'positive_percent', 'genres', 'tags',
        'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
    ])

    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)
    df = df.dropna(subset=['current_price', 'discounted_price'])

    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

    df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',')])
    df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';')])

    df['review'] = df.get('review', '').fillna('')
    tfidf_matrix = tfidf.transform(df['review'])
    df['review_score'] = svd.transform(tfidf_matrix).flatten()

    df = competitor_transformer.transform(df)

    genres_encoded = pd.DataFrame(mlb_genres.transform(df['genres']), columns=genre_cols, index=df.index)
    tags_encoded = pd.DataFrame(mlb_tags.transform(df['tags']), columns=tag_cols, index=df.index)

    df = pd.concat([df, genres_encoded, tags_encoded], axis=1)

    return df

def prep_db():
    with get_connection(dbname='postgres', autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if not res.fetchall():
            conn.execute("CREATE DATABASE test;")
    with get_connection(dbname='test', autocommit=True) as conn:
        conn.execute(CREATE_TABLE_SQL)

def get_current_batch(i):
    full_data = pd.read_csv("data/combined4.csv")
    full_data = full_data.sort_values(by='days_after_publish')
    day_i = full_data[full_data['days_after_publish'] == i].copy()
    if day_i.empty:
        return None

    processed_batch = preprocess_data(day_i)
    X = processed_batch[all_features]
    processed_batch['prediction'] = model.predict(X)

    return processed_batch

def run_drift_report(ref_data, curr_data):
    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])
    report.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)
    return report.as_dict()

def insert_metrics_to_db(metrics_dict, timestamp, cursor):
    prediction_drift = metrics_dict['metrics'][0]['result']['drift_score']
    num_drifted_columns = metrics_dict['metrics'][1]['result']['number_of_drifted_columns']
    share_missing = metrics_dict['metrics'][2]['result']['current']['share_of_missing_values']

    cursor.execute(
        """
        INSERT INTO discount_model_metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values)
        VALUES (%s, %s, %s, %s)
        """,
        (timestamp, prediction_drift, num_drifted_columns, share_missing)
    )

if __name__ == "__main__":
    reference_raw = pd.read_csv("data/combined4.csv")
    reference_data = preprocess_data(reference_raw)

    prep_db()
    last_send = datetime.datetime.now(pytz.UTC) - datetime.timedelta(seconds=SEND_TIMEOUT)
    with get_connection(dbname='test', autocommit=True) as conn:
        for i in range(0, 30):
            with conn.cursor() as cursor:
                batch = get_current_batch(i)
                if batch is None:
                    logging.warning(f"No data found for batch {i}. Skipping.")
                    continue

                drift_report = run_drift_report(reference_data, batch)
                insert_metrics_to_db(drift_report, last_send, cursor)
                logging.info(f"Drift logged for batch {i}")

            now = datetime.datetime.now(pytz.UTC)
            elapsed = (now - last_send).total_seconds()
            if elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - elapsed)
            last_send += datetime.timedelta(seconds=SEND_TIMEOUT)
