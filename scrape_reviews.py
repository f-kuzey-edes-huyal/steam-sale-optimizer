import nest_asyncio
nest_asyncio.apply()

import asyncio
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd

# Load game IDs from CSV
csv_path = "steam_recent_games.csv"  # Adjust path as needed
df = pd.read_csv(csv_path)
game_ids = df["game_id"].dropna().astype(str).unique().tolist()

async def scrape_reviews_for_game(page, game_id):
    url = f"https://store.steampowered.com/app/{game_id}"
    await page.goto(url)

    try:
        await page.wait_for_selector(".user_reviews", timeout=60000)
        reviews_text = await page.inner_text(".user_reviews")
        return {"game_id": game_id, "reviews": reviews_text}
    except PlaywrightTimeoutError:
        print(f"⚠️ Timeout waiting for reviews on game {game_id}")
        return {"game_id": game_id, "reviews": None}
    except Exception as e:
        print(f"⚠️ Failed to scrape {game_id}: {e}")
        return {"game_id": game_id, "reviews": None}

async def run_scrape(game_ids):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        results = []
        for game_id in game_ids:
            print(f"🔍 Scraping reviews for Game ID: {game_id}")
            result = await scrape_reviews_for_game(page, game_id)
            results.append(result)

        await browser.close()

        reviews_df = pd.DataFrame(results)
        reviews_df.to_json('steam_data/reviews.json', orient='records', indent=2)
        print("✅ Data collection completed and saved in steam_data/reviews.json")

# Entry point for the script
if __name__ == "__main__":
    asyncio.run(run_scrape(game_ids))

