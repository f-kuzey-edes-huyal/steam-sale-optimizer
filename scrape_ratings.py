import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def get_steam_api_reviews(appid):
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&review_type=all&purchase_type=all&language=all"
    try:
        response = requests.get(url)
        data = response.json()
        summary = data.get("query_summary", {})
        total_reviews = summary.get("total_reviews", 0)
        positive_reviews = summary.get("total_positive", 0)
        negative_reviews = summary.get("total_negative", 0)
        if total_reviews > 0:
            rating_percent = int((positive_reviews / total_reviews) * 100)
        else:
            rating_percent = 0
        return total_reviews, positive_reviews, negative_reviews, rating_percent
    except Exception as e:
        print(f"[API ERROR] appid={appid}: {e}")
        return 0, 0, 0, 0

def scrape_steamdb_data(appid, driver):
    url = f"https://steamdb.info/app/{appid}/"
    driver.get(url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Followers
    followers = 0
    # Try to find followers count - SteamDB changes page structure sometimes
    followers_div = soup.find("div", class_="app-followers")
    if followers_div:
        followers_text = followers_div.text.strip().replace(",", "")
        if followers_text.isdigit():
            followers = int(followers_text)

    # Peak players
    peak_players = 0
    peak_tag = soup.find("td", string="Peak players")
    if peak_tag:
        peak_value_tag = peak_tag.find_next_sibling("td")
        if peak_value_tag:
            peak_str = peak_value_tag.text.strip().replace(",", "")
            try:
                peak_players = int(peak_str)
            except:
                peak_players = 0

    return followers, peak_players

def main():
    # Read game IDs from CSV
    df_games = pd.read_csv("steam_recent_games.csv")
    game_ids = df_games['game_id'].dropna().astype(int).tolist()

    driver = create_driver()
    results = []

    for idx, appid in enumerate(game_ids):
        print(f"Processing {idx+1}/{len(game_ids)}: appid={appid}")

        total_reviews, positive, negative, rating_pct = get_steam_api_reviews(appid)
        followers, peak_players = scrape_steamdb_data(appid, driver)

        results.append({
            "appid": appid,
            "total_reviews": total_reviews,
            "positive_reviews": positive,
            "negative_reviews": negative,
            "rating_percent": rating_pct,
            "followers": followers,
            "peak_players": peak_players
        })

        time.sleep(1)  # polite pause

    driver.quit()

    df_results = pd.DataFrame(results)
    df_results.to_csv("steam_detailed_info.csv", index=False)
    print("✅ Saved detailed info to steam_detailed_info.csv")

if __name__ == "__main__":
    main()
