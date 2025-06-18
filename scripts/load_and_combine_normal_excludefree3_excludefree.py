import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import re

load_dotenv()

def parse_owners(owners_str):
    try:
        if pd.isna(owners_str):
            return None, None, None
        parts = owners_str.replace(",", "").split("..")
        if len(parts) != 2:
            print(f"Skipping invalid format: {owners_str}")
            return None, None, None
        low = int(parts[0].strip())
        high = int(parts[1].strip())
        log_mean = np.exp((np.log(low) + np.log(high)) / 2)
        return low, high, log_mean
    except Exception as e:
        print(f"Error parsing owners_str '{owners_str}': {e}")
        return None, None, None

def clean_review(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s.,;:\-!?\']+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_csv_to_postgres_and_export():
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "yourpassword")
    db = os.getenv("POSTGRES_DB", "steamdb")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(db_url)

    print("Loading CSV files...")
    reviews = pd.read_csv('data/reviews.csv')
    steam_api = pd.read_csv('data/steamspy_data.csv')
    steamdata = pd.read_csv('data/steamdata.csv')

    steamdata['release_date'] = pd.to_datetime(steamdata['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')

    print("CSV file shapes:")
    print("reviews:", reviews.shape)
    print("steam_api:", steam_api.shape)
    print("steamdata:", steamdata.shape)

    reviews.to_sql('reviews', engine, if_exists='replace', index=False)
    steam_api.to_sql('steam_api', engine, if_exists='replace', index=False)
    steamdata.to_sql('steamdata', engine, if_exists='replace', index=False)
    print("Uploaded CSVs to PostgreSQL")

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS combined;"))
        conn.execute(text("DROP TABLE IF EXISTS combined_step1;"))
        conn.execute(text("DROP TABLE IF EXISTS aggregated_reviews;"))

        conn.execute(text("""
            CREATE TABLE aggregated_reviews AS
            SELECT
                game_id,
                STRING_AGG(review, ' ') AS all_reviews
            FROM reviews
            GROUP BY game_id;
        """))
        print("Aggregated reviews per game_id")

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
        print("Step 1: Joined steamdata, steam_api, and aggregated reviews")

        conn.execute(text("""
            ALTER TABLE combined_step1
            ADD COLUMN days_after_publish INT;
        """))

        conn.execute(text("""
            UPDATE combined_step1
            SET days_after_publish = DATE_PART('day', NOW() - TO_DATE(release_date, 'YYYY-MM-DD'))
            WHERE release_date IS NOT NULL;
        """))
        print("Step 2: Added days_after_publish column")

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
        print("Step 3: Final combined table created")

    combined_df = pd.read_sql("SELECT * FROM combined", engine)
    print("Loaded combined table into DataFrame")
    print("combined_df shape:", combined_df.shape)

    print("Cleaning and parsing 'review' and 'owners' columns...")
    combined_df['review'] = combined_df['review'].apply(clean_review)

    combined_df[['owner_min', 'owner_max', 'owners_log_mean']] = combined_df['owners'].apply(
        lambda x: pd.Series(parse_owners(x))
    )

    # Exclude games marked as 'free' in current_price
    combined_df = combined_df[~combined_df['current_price'].astype(str).str.lower().str.contains('free')]

    output_path = 'data/combined4.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Exported combined DataFrame to '{output_path}'")
    print(combined_df[['owners', 'owner_min', 'owner_max', 'owners_log_mean']].head())

if __name__ == "__main__":
    load_csv_to_postgres_and_export()
