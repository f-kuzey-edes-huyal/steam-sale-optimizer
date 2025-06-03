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
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def is_free(price_text):
    price_text = price_text.strip()
    return price_text.lower().startswith("free") or price_text == ""

def scrape_steam_top_sellers(target_games=2300):
    driver = create_driver()
    base_url = "https://store.steampowered.com/search/?filter=topsellers&ignore_preferences=1"
    driver.get(base_url)
    time.sleep(3)

    collected = []
    page = 1

    while len(collected) < target_games:
        print(f"📄 Page {page}: {len(collected)} / {target_games} games collected.")
        rows = driver.find_elements(By.CSS_SELECTOR, "a.search_result_row")

        for row in rows:
            if len(collected) >= target_games:
                break

            try:
                href = row.get_attribute("href")
                game_id_match = re.search(r"/app/(\d+)", href)
                if not game_id_match:
                    continue
                game_id = int(game_id_match.group(1))

                name = row.find_element(By.CSS_SELECTOR, ".title").text.strip()
                release_date = row.find_element(By.CSS_SELECTOR, ".search_released").text.strip()
                price_text = row.find_element(By.CSS_SELECTOR, ".search_price").text.strip()

                # Skip free games
                if is_free(price_text):
                    continue

                # Default review info
                total_reviews = 0
                positive_percent = None

                try:
                    review_elem = row.find_element(By.CSS_SELECTOR, ".search_review_summary")
                    tooltip = review_elem.get_attribute("data-tooltip-html")
                    if tooltip:
                        review_match = re.search(r"([\d,]+) user reviews", tooltip)
                        percent_match = re.search(r"(\d+)%", tooltip)

                        if review_match:
                            total_reviews = int(review_match.group(1).replace(",", ""))
                        if percent_match:
                            positive_percent = int(percent_match.group(1))
                except:
                    pass  # keep default values if missing

                # Filter based on reviews
                if total_reviews < 500:
                    continue
                if positive_percent is not None and positive_percent >= 90:
                    continue

                collected.append({
                    "game_id": game_id,
                    "name": name,
                    "release_date": release_date,
                    "total_reviews": total_reviews,
                    "positive_percent": positive_percent,
                    "price_text": price_text,
                })

            except Exception as e:
                print(f"⚠️ Error parsing row: {e}")
                continue

        # Next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "a.pagebtn:not(.disabled):nth-last-child(1)")
            next_btn.click()
            time.sleep(2)
            page += 1
        except:
            print("No more pages or next button failed.")
            break

    driver.quit()
    return pd.DataFrame(collected)

if __name__ == "__main__":
    df = scrape_steam_top_sellers(target_games=2300)
    print(f"Scraped {len(df)} valid paid games.")
    df.to_csv("steam_data/filtered_steam_games.csv", index=False)
