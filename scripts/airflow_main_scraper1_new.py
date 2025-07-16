from steam_scraper_air_new import scrape_steam_top_games
from steamspy_scrape_air_new import fetch_and_save_steamspy_data
from review_scraper_air_new import get_reviews

import pandas as pd
import os
import logging


def scrape_steam_data():
    try:
        # Step 1: Scrape top Steam games
        df = scrape_steam_top_games(max_games=200)
        if df.empty:
            logging.warning("No games scraped. Exiting task.")
            return

        game_ids = df['game_id'].tolist()

        # Step 2: Save review data for each game
        all_reviews = []
        for appid in game_ids:
            try:
                print(f"Fetching reviews for AppID: {appid}")
                reviews = get_reviews(appid, num_reviews=5)
                for review in reviews:
                    review['game_id'] = appid
                all_reviews.extend(reviews)
            except Exception as e:
                logging.error(f"Failed to fetch reviews for {appid}: {e}")

        reviews_df = pd.DataFrame(all_reviews)
        os.makedirs("data", exist_ok=True)
        reviews_df.to_csv("data/reviews_apache.csv", index=False)
        print(f"Saved reviews for {len(reviews_df)} entries.")

        # Step 3: Fetch and save SteamSpy data
        fetch_and_save_steamspy_data(game_ids, save_path="data/steam_api_apache.csv")
        print("SteamSpy data saved.")

    except Exception as e:
        logging.exception("Error in scrape_steam_data pipeline")
        raise
