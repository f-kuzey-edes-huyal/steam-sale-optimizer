from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from dateutil import parser
import time
import pandas as pd
import re
from datetime import datetime
import random
import os

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def parse_release_year(release_str):
    try:
        dt = parser.parse(release_str, fuzzy=True)
        return dt.year
    except Exception:
        return None

def scrape_steam_top_10000():
    driver = create_driver()
    games = []

    for page in range(1, 401):  # 100 pages × 25 games = 2500 max
        print(f"Scraping page {page}...")
        url = f"https://store.steampowered.com/search/?filter=topsellers&ignore_preferences=1&page={page}"
        driver.get(url)
        time.sleep(random.uniform(2, 4))

        rows = driver.find_elements(By.CSS_SELECTOR, "a.search_result_row")
        
        for row in rows:
            try:
                href = row.get_attribute("href")
                match = re.search(r"/app/(\d+)", href)
                if not match:
                    continue
                game_id = int(match.group(1))

                name = row.find_element(By.CSS_SELECTOR, ".title").text.strip()
                release_date_str = row.find_element(By.CSS_SELECTOR, ".search_released").text.strip()
                release_year = parse_release_year(release_date_str)

                if not release_year: 
                   
                    continue

                try:
                    review_elem = row.find_element(By.CSS_SELECTOR, ".search_review_summary")
                    review_title = review_elem.get_attribute("data-tooltip-html")
                    match_reviews = re.search(r"([\d,]+) user reviews", review_title)
                    match_pos = re.search(r"(\d+)% of the", review_title)
                    total_reviews = int(match_reviews.group(1).replace(",", "")) if match_reviews else None
                    positive_pct = int(match_pos.group(1)) if match_pos else None
                except:
                    total_reviews = None
                    positive_pct = None

                games.append({
                    "game_id": game_id,
                    "name": name,
                    "release_date": release_date_str,
                    "release_year": release_year,
                    "total_reviews": total_reviews,
                    "positive_percent": positive_pct
                })

                if len(games) >= 2500:
                    break

            except Exception as e:
                print("Error parsing game:", e)

        if len(games) >= 2500:
            break

    driver.quit()

    # Save data
    os.makedirs("steam_data", exist_ok=True)
    df = pd.DataFrame(games)
    df.to_csv("steam_data/steam_2020_to_2025_games.csv", index=False)
    print("Scraping complete. Total games collected:", len(df))
    return df

if __name__ == "__main__":
    df = scrape_steam_top_10000()
    print(df.head())
