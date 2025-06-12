# airflow_main_scraper1.py

from steam_scraper_air import scrape_steam_top_games
from steamspy_scrape_air import fetch_and_save_steamspy_data
from review_scraper_air import get_reviews
import pandas as pd
import requests
import os

def scrape_steam_data():
    df = scrape_steam_top_games(max_games=2)
    game_ids = df['game_id'].tolist()

    



    all_reviews = []
    for appid in game_ids:
        print(f"Fetching reviews for AppID: {appid}")
        reviews = get_reviews(appid, num_reviews=5)
        for review in reviews:
            review['game_id'] = appid
        all_reviews.extend(reviews)

    reviews_df = pd.DataFrame(all_reviews)
    os.makedirs("data", exist_ok=True)
    reviews_df.to_csv("data/reviews.csv", index=False)
    fetch_and_save_steamspy_data(game_ids, save_path="data/steam_api.csv")
    print("Review scraping complete.")
