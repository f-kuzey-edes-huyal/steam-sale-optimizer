from temp_scripts.steamdb_scrape import get_recent_games

df = get_recent_games(pages=5)
print(f"[INFO] Total recent games collected: {len(df)} 🎮")
print(df.head())

df.to_csv("steam_recent_games.csv", index=False)
print("[✅] Data saved to 'steam_recent_games.csv' 📁")
