import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import re
import logging

# Load environment variables from .env file
load_dotenv()

def parse_owners(owners_str):
    try:
        if pd.isna(owners_str):
            return None, None, None
        parts = owners_str.replace(",", "").split("..")
        if len(parts) != 2:
            logging.warning(f"Skipping invalid format: {owners_str}")
            return None, None, None
        low = int(parts[0].strip())
        high = int(parts[1].strip())
        log_mean = np.exp((np.log(low) + np.log(high)) / 2)
        return low, high, log_mean
    except Exception as e:
        logging.error(f"Error parsing owners_str '{owners_str}': {e}")
        return None, None, None

def clean_review(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s.,;:\-!?\']+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_csv_to_postgres_and_export():
    try:
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        db = os.getenv("POSTGRES_DB")
        db_url = f"postgresql://{user}:{password}@postgres:5432/{db}"
        engine = create_engine(db_url)

        # Load CSV files
        reviews = pd.read_csv('/opt/airflow/data/reviews_apache.csv')
        steam_api = pd.read_csv('/opt/airflow/data/steam_api_apache.csv')
        steamdata = pd.read_csv('/opt/airflow/data/steamdata_apache.csv')

        steamdata['release_date'] = pd.to_datetime(
            steamdata['release_date'], errors='coerce'
        ).dt.strftime('%Y-%m-%d')

        # Use engine.connect() to get Connection for pandas.to_sql
        with engine.connect() as conn:
            reviews.to_sql('reviews', con=conn, if_exists='replace', index=False)
            steam_api.to_sql('steam_api', con=conn, if_exists='replace', index=False)
            steamdata.to_sql('steamdata', con=conn, if_exists='replace', index=False)

        print("Uploaded CSVs to PostgreSQL")

        # Use a transaction for your SQL commands
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS combined, combined_step1, aggregated_reviews;"))

            conn.execute(text("""
                CREATE TABLE aggregated_reviews AS
                SELECT game_id, STRING_AGG(review, ' ') AS all_reviews
                FROM reviews
                GROUP BY game_id;
            """))

            conn.execute(text("""
                CREATE TABLE combined_step1 AS
                SELECT
                    sd.game_id,
                    sd.name,
                    sd.release_date,
                    sd.total_reviews,
                    sd.positive_percent,
                    sd.genres,
                    sd.tags,
                    sd.current_price,
                    sd.discounted_price,
                    sa.owners,
                    ar.all_reviews AS review
                FROM steamdata sd
                LEFT JOIN steam_api sa ON sd.game_id = sa.game_id
                LEFT JOIN aggregated_reviews ar ON sd.game_id = ar.game_id;
            """))

            conn.execute(text("ALTER TABLE combined_step1 ADD COLUMN days_after_publish INT;"))

            conn.execute(text("""
                UPDATE combined_step1
                SET days_after_publish = DATE_PART('day', NOW() - TO_DATE(release_date, 'YYYY-MM-DD'))
                WHERE release_date IS NOT NULL;
            """))

            conn.execute(text("""
                CREATE TABLE combined AS
                SELECT
                    game_id,
                    name,
                    release_date,
                    total_reviews,
                    positive_percent,
                    genres,
                    tags,
                    current_price,
                    discounted_price,
                    owners,
                    days_after_publish,
                    review
                FROM combined_step1;
            """))

            print("Created final 'combined' table")

        # Read final combined table and process
        combined_df = pd.read_sql("SELECT * FROM combined", con=engine)

        combined_df['review'] = combined_df['review'].apply(clean_review)
        combined_df[['owner_min', 'owner_max', 'owners_log_mean']] = combined_df['owners'].apply(
            lambda x: pd.Series(parse_owners(x))
        )

        # Exclude free games
        combined_df = combined_df[
            ~combined_df['current_price'].astype(str).str.lower().str.contains('free')
        ]

        combined_df.to_csv('/opt/airflow/data/combined_outputnew.csv', index=False)
        print("Exported final DataFrame to /opt/airflow/data/combined_outputnew.csv")

    except Exception as e:
        logging.exception("Error in load_csv_to_postgres_and_export")
        raise

if __name__ == "__main__":
    load_csv_to_postgres_and_export()
