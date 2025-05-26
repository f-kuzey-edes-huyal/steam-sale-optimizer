from temp_scripts.steamdb_scrape import get_recent_games
import os
import time
import pandas as pd


# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    df = get_recent_games(pages=5)  # You can increase to 10+
    print(f"[INFO] Total recent games collected: {len(df)}")
    print(df.head())

    # Save to CSV
    output_path = "recent_steam_games.csv"
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved to file: {output_path}")
