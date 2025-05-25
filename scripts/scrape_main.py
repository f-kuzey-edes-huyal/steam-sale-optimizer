import pandas as pd
import sys
import os
print(os.getcwd())
# Add current directory to Python path
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from temp_scripts.steamdb_scrape import get_recent_games




df = get_recent_games(pages=10)

for i, row in df.iterrows():
    dev, pub, price = get_game_details(row["game_id"])
    df.at[i, "developer"] = dev
    df.at[i, "publisher"] = pub
    df.at[i, "base_price"] = price
    time.sleep(1)

# Save results
os.makedirs("steam_data", exist_ok=True)
df.to_json("steam_data/recent_games.json", orient="records", indent=2)
print("✅ Data saved to 'steam_data/recent_games.json'")
