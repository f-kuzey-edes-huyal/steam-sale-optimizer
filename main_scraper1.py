from steam_scraper import scrape_steam_top_games
from steamspy_scrape import fetch_and_save_steamspy_data
from review_scraper import get_reviews

import pandas as pd
import os

def main():
    df = scrape_steam_top_games(max_games=100)  # Adjust as needed
    game_ids = df['game_id'].tolist()

    # Fetch and save SteamSpy data separately
    fetch_and_save_steamspy_data(game_ids, save_path="data/steamspy_data.csv")

    all_reviews = []
    for appid in game_ids:
        print(f"Fetching reviews for AppID: {appid}")
        reviews = get_reviews(appid, num_reviews=20)
        for review in reviews:
            review['game_id'] = appid
        all_reviews.extend(reviews)

    # Save reviews
    reviews_df = pd.DataFrame(all_reviews)
    os.makedirs("data", exist_ok=True)
    reviews_df.to_csv("data/steam_reviews.csv", index=False)
    print("Review scraping complete.")

if __name__ == "__main__":
    main()