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

from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping
from train_optuna_hyperparameter_mlflow_reviews_competitor_pricing_change_criterion_mean_absolute import CompetitorPricingTransformer

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


model = joblib.load('models/discount_model_pipeline_small.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
svd = joblib.load('models/svd_transform.pkl')
mlb_genres = joblib.load('models/mlb_genres.pkl')
mlb_tags = joblib.load('models/mlb_tags.pkl')
comp_tr = joblib.load('models/competitor_pricing_transformer.pkl')

RAW_FEATURES = [
    'total_reviews',
    'positive_percent',
    'current_price',
    'discounted_price',
    'owners_log_mean',
    'days_after_publish',
    'review',
    'genres',
    'tags'
]

MODEL_FEATURES = [
    'total_reviews',
    'positive_percent',
    'current_price',
    'discounted_price',
    'owners_log_mean',
    'days_after_publish',
    'review_score',
    'competitor_pricing'
] + list(mlb_genres.classes_) + list(mlb_tags.classes_)

DRIFT_FEATURES = list(dict.fromkeys(MODEL_FEATURES))

column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=DRIFT_FEATURES,
    target=None
)

CREATE_TABLE_SQL = """
DROP TABLE IF EXISTS discount_model_metrics;
CREATE TABLE discount_model_metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT
)
"""


def parse_price(x):
    try:
        return float(str(x).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan


def process_genres(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return list(set(g.strip() for g in x.split(',') if g.strip()))
    return []


def process_tags(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return list(set(t.strip() for t in x.split(';') if t.strip()))
    return []


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=[
        'total_reviews', 'positive_percent', 'genres', 'tags',
        'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
    ])

    df['current_price'] = df['current_price'].apply(parse_price)
    df['discounted_price'] = df['discounted_price'].apply(parse_price)
    df = df.dropna(subset=['current_price', 'discounted_price'])

    df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

    # Apply fixed robust processing functions to genres and tags
    df['genres'] = df['genres'].apply(process_genres)
    df['tags'] = df['tags'].apply(process_tags)

    valid_genres = set(mlb_genres.classes_)
    valid_tags = set(mlb_tags.classes_)

    df['genres'] = df['genres'].apply(lambda genres: [g for g in genres if g in valid_genres])
    df['tags'] = df['tags'].apply(lambda tags: [t for t in tags if t in valid_tags])

    df['review'] = df.get('review', '').fillna('')

    # Transform reviews to review_score
    tfm = tfidf.transform(df['review'])
    df['review_score'] = svd.transform(tfm).flatten()

    # Transform competitor pricing feature
    comp_tr_out = comp_tr.transform(df)
    if not isinstance(comp_tr_out, pd.DataFrame):
        comp_tr_out = pd.DataFrame(comp_tr_out, index=df.index, columns=['competitor_pricing'])
    else:
        if comp_tr_out.shape[1] == 1:
            comp_tr_out.columns = ['competitor_pricing']

    df = pd.concat([df, comp_tr_out], axis=1)

    logging.info(f"Preprocessing {len(df)} rows for multilabel binarization.")

    genres_transformed = mlb_genres.transform(df['genres'])
    tags_transformed = mlb_tags.transform(df['tags'])

    if genres_transformed.shape[0] != len(df):
        raise ValueError(f"Genres transform rows ({genres_transformed.shape[0]}) != dataframe rows ({len(df)})")

    if tags_transformed.shape[0] != len(df):
        raise ValueError(f"Tags transform rows ({tags_transformed.shape[0]}) != dataframe rows ({len(df)})")

    genres_encoded = pd.DataFrame(genres_transformed, columns=mlb_genres.classes_, index=df.index)
    tags_encoded = pd.DataFrame(tags_transformed, columns=mlb_tags.classes_, index=df.index)

    df = pd.concat([df, genres_encoded, tags_encoded], axis=1)

    df = df.drop(columns=['genres', 'tags', 'review'], errors='ignore')

    df = df.loc[:, ~df.columns.duplicated()]

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
            cur.execute(CREATE_TABLE_SQL)


def get_batch(i: int):
    df_raw = pd.read_csv("data/combined4.csv").sort_values('days_after_publish')
    batch_raw = df_raw[df_raw['days_after_publish'] == i]
    if batch_raw.empty:
        return None

    batch_raw = batch_raw.dropna(subset=RAW_FEATURES)

    batch_processed = preprocess(batch_raw)

    batch_processed = batch_processed.dropna(subset=MODEL_FEATURES)

    batch_processed['prediction'] = model.predict(batch_processed)

    return batch_processed


def run_drift(ref: pd.DataFrame, cur: pd.DataFrame):
    for df in (ref, cur):
        for col in ['genres', 'tags', 'review']:
            df.pop(col, None)
        df.drop_duplicates(axis=1, inplace=True)

    for df in (ref, cur):
        for col in DRIFT_FEATURES + ['prediction']:
            if col not in df.columns:
                df[col] = 0

    expected_cols = DRIFT_FEATURES + ['prediction']

    for df in (ref, cur):
        df_reindexed = df.reindex(columns=expected_cols, fill_value=0)
        df.drop(df.columns, axis=1, inplace=True)
        for col in df_reindexed.columns:
            df[col] = df_reindexed[col]

    rpt = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    rpt.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

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
        reference = preprocess(reference_raw)
        reference = reference.dropna(subset=MODEL_FEATURES)
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
