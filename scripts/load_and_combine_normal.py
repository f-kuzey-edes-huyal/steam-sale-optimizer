import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

def parse_owners(owners_str):
    try:
        # Example format: "2,000,000 .. 5,000,000"
        parts = owners_str.replace(",", "").split("..")
        if len(parts) != 2:
            return None, None, None
        low = int(parts[0].strip())
        high = int(parts[1].strip())
        log_mean = np.exp((np.log(low) + np.log(high)) / 2)
        return low, high, log_mean
    except:
        return None, None, None

def load_csv_to_postgres_and_export():
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "yourpassword")
    db = os.getenv("POSTGRES_DB", "steamdb")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")

    db_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(db_url)

    reviews = pd.read_csv('data/reviews.csv')
    steam_api = pd.read_csv('data/steam_api.csv')
    steamdata = pd.read_csv('data/steamdata.csv')

    reviews.to_sql('reviews', engine, if_exists='replace', index=False)
    steam_api.to_sql('steam_api', engine, if_exists='replace', index=False)
    steamdata.to_sql('steamdata', engine, if_exists='replace', index=False)
    print("Uploaded CSVs to PostgreSQL")

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS combined;"))

        conn.execute(text("""
            CREATE TABLE combined_step1 AS
            SELECT
                sd.game_id,
                sd.name,
                sd.release_date,
                sd.total_reviews,
                sd.positive_percent_genres,
                sd.tags,
                sd.current_price,
                sd.discounted_price,
                sa.owners,
                r.review
            FROM steamdata sd
            LEFT JOIN steam_api sa ON sd.game_id = sa.game_id
            LEFT JOIN reviews r ON sd.game_id = r.game_id;
        """))
        print("Step 1: Joined steamdata, steam_api, reviews")

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
                positive_percent_genres,
                tags,
                current_price,
                discounted_price,
                owners,
                days_after_publish,
                review
            FROM combined_step1;
        """))
        print("Step 3: Final combined table created")

    # Export combined to DataFrame
    combined_df = pd.read_sql("SELECT * FROM combined", engine)

    # Process owners column to create owner_min, owner_max, owners_log_mean
    combined_df[['owner_min', 'owner_max', 'owners_log_mean']] = combined_df['owners'].apply(
        lambda x: pd.Series(parse_owners(x))
    )

    combined_df.to_csv('data/combined.csv', index=False)
    print("Exported combined.csv")
