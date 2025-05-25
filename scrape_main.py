from temp_scripts.steamdb_scrape import get_recent_games, get_game_details
import os
import time
import pandas as pd


df = get_recent_games(pages=10)

for i, row in df.iterrows():
    dev, pub, price = get_game_details(row['game_id'])
    df.at[i, 'developer'] = dev
    df.at[i, 'publisher'] = pub
    df.at[i, 'base_price'] = price
    time.sleep(1)  # Be polite with requests

print(df.head())

# Save to JSON
df.to_json('steam_data/recent_games.json', orient='records', indent=2)
print("✅ Data saved.")
