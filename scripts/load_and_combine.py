
import os
import pandas as pd
from sqlalchemy import create_engine, text

def load_csv_to_postgres_and_export():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    db = os.getenv("POSTGRES_DB")

    db_url = f"postgresql://{user}:{password}@postgres:5432/{db}"
    engine = create_engine(db_url)

    reviews = pd.read_csv('/opt/airflow/data/reviews.csv')
    steam_api = pd.read_csv('/opt/airflow/data/steam_api.csv')
    steamdata = pd.read_csv('/opt/airflow/data/steamdata.csv')

    reviews.to_sql('reviews', engine, if_exists='replace', index=False)
    steam_api.to_sql('steam_api', engine, if_exists='replace', index=False)
    steamdata.to_sql('steamdata', engine, if_exists='replace', index=False)
    print("✅ Uploaded CSVs to PostgreSQL")

    join_query = """
    DROP TABLE IF EXISTS combined;
    CREATE TABLE combined AS
    SELECT
        r.review,
        r.game_id,
        s.name AS game_name,
        s.genres,
        a.owners
    FROM reviews r
    LEFT JOIN steamdata s ON r.game_id = s.game_id
    LEFT JOIN steam_api a ON r.game_id = a.game_id;
    """

    with engine.connect() as conn:
        conn.execute(text(join_query))
        print("✅ Created combined table")

    combined_df = pd.read_sql("SELECT * FROM combined", engine)
    combined_df.to_csv('/opt/airflow/data/combined_output.csv', index=False)
    print("✅ Exported final CSV")
