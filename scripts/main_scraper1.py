from steam_scraper import scrape_steam_top_games
from review_scraper import get_reviews
import pandas as pd
import os

def main():
    df = scrape_steam_top_games(max_games=5000)  # Adjust as needed
    all_reviews = []
    for appid in df['game_id']:
        print(f"Fetching reviews for AppID: {appid}")
        reviews = get_reviews(appid, num_reviews=20)
        for review in reviews:
            review['game_id'] = appid
        all_reviews.extend(reviews)
    # Save reviews
    reviews_df = pd.DataFrame(all_reviews)
    reviews_df.to_csv("data/raw/steam_reviews.csv", index=False)
    print("Review scraping complete.")

if __name__ == "__main__":
    main()
