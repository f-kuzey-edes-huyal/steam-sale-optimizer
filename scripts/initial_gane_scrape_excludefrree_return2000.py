from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import re

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def parse_review_text(review_text):
    pattern = r"([A-Za-z\s]+)\s*\(([\d,]+) reviews\)"
    match = re.search(pattern, review_text)
    if match:
        rating_summary = match.group(1).strip()
        total_reviews = int(match.group(2).replace(",", ""))
        return rating_summary, total_reviews
    else:
        return None, 0

def scrape_steam_top_sellers(max_games=2500):
    driver = create_driver()
    base_url = "https://store.steampowered.com/search/?filter=topsellers&ignore_preferences=1"
    driver.get(base_url)
    time.sleep(3)

    games = []
    page = 1

    while len(games) < max_games:
        print(f"Scraping page {page}... Total games collected: {len(games)}")
        rows = driver.find_elements(By.CSS_SELECTOR, "a.search_result_row")
        
        for row in rows:
            try:
                href = row.get_attribute("href")
                match = re.search(r"/app/(\d+)", href)
                if not match:
                    continue
                game_id = int(match.group(1))

                name = row.find_element(By.CSS_SELECTOR, ".title").text.strip()
                release_date = row.find_element(By.CSS_SELECTOR, ".search_released").text.strip()

                try:
                    review_elem = row.find_element(By.CSS_SELECTOR, ".search_review_summary")
                    review_title = review_elem.get_attribute("data-tooltip-html")
                    positive_pct = None
                    total_reviews = None
                    match_reviews = re.search(r"(\d[\d,]*) user reviews", review_title)
                    match_pos = re.search(r"(\d+)% of the", review_title)
                    if match_reviews:
                        total_reviews = int(match_reviews.group(1).replace(",", ""))
                    if match_pos:
                        positive_pct = int(match_pos.group(1))
                except:
                    total_reviews = 0
                    positive_pct = None

                if total_reviews is None or total_reviews < 500:
                    continue
                if positive_pct is not None and positive_pct >= 90:
                    continue

                games.append({
                    "game_id": game_id,
                    "name": name,
                    "release_date": release_date,
                    "total_reviews": total_reviews,
                    "positive_percent": positive_pct
                })

                if len(games) >= max_games:
                    break
            except Exception as e:
                print("Error parsing a game row:", e)
                continue

        try:
            next_button = driver.find_element(By.CSS_SELECTOR, "a.pagebtn:nth-child(4)")
            if "disabled" in next_button.get_attribute("class"):
                break
            next_button.click()
            time.sleep(3)
            page += 1
        except Exception as e:
            print("No more pages or error:", e)
            break

    driver.quit()
    return pd.DataFrame(games)

def scrape_pricing_for_games(df_games):
    driver = create_driver()
    pricing_list = []

    for idx, row in df_games.iterrows():
        game_id = row["game_id"]
        url = f"https://store.steampowered.com/app/{game_id}/"
        driver.get(url)
        time.sleep(3)

        try:
            discount_price = None
            original_price = None
            discount_pct = 0

            try:
                discount_price = driver.find_element(By.CSS_SELECTOR, "div.discount_final_price").text.strip()
                original_price = driver.find_element(By.CSS_SELECTOR, "div.discount_original_price").text.strip()
                op = float(re.sub(r"[^\d.]", "", original_price))
                dp = float(re.sub(r"[^\d.]", "", discount_price))
                discount_pct = round((1 - dp / op) * 100, 2)
            except:
                price_elem = driver.find_element(By.CSS_SELECTOR, "div.game_purchase_price.price")
                original_price = price_elem.text.strip()

                # Skip free games
                if "Free" in original_price:
                    print(f"Skipping free game {game_id}")
                    continue

                discount_price = original_price
                discount_pct = 0

            pricing_list.append({
                "game_id": game_id,
                "original_price": original_price,
                "discount_price": discount_price,
                "discount_percent": discount_pct
            })
        except Exception as e:
            print(f"Failed to get pricing for game {game_id}: {e}")
            pricing_list.append({
                "game_id": game_id,
                "original_price": None,
                "discount_price": None,
                "discount_percent": None
            })

    driver.quit()
    return pd.DataFrame(pricing_list)

if __name__ == "__main__":
    df_games = scrape_steam_top_sellers(max_games=2500)
    print(df_games.head())
    df_games.to_csv("steam_data/filtered_steam_games.csv", index=False)

    df_pricing = scrape_pricing_for_games(df_games)
    print(df_pricing.head())
    df_pricing.to_csv("steam_data/steam_pricing.csv", index=False)

    print("Scraping complete!")
