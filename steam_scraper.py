from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from dateutil import parser
import time
import pandas as pd
import re
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

def get_additional_game_info(game_url, driver):
    driver.get(game_url)
    time.sleep(random.uniform(2, 4))

    info = {
        "developer": "",
        "publisher": "",
        "genres": "",
        "platforms": "",
        "current_price": "",
        "discounted_price": "",
        "discount_percent": ""
    }

    try:
        # Developer & Publisher
        dev_blocks = driver.find_elements(By.CSS_SELECTOR, ".dev_row")
        for row in dev_blocks:
            try:
                label = row.find_element(By.CLASS_NAME, "subtitle").text
                value = row.find_element(By.CLASS_NAME, "summary").text
                if "Developer" in label:
                    info["developer"] = value.strip()
                elif "Publisher" in label:
                    info["publisher"] = value.strip()
            except:
                continue

        # Genres
        try:
            genre_elems = driver.find_elements(By.CSS_SELECTOR, "a[href*='genre']")
            genres = [g.text.strip() for g in genre_elems if g.text.strip()]
            info["genres"] = ", ".join(genres)
        except:
            pass

        # Platforms
        try:
            platforms = driver.find_elements(By.CSS_SELECTOR, ".game_area_purchase_platform .platform_img")
            platform_list = []
            for p in platforms:
                cls = p.get_attribute("class")
                if "win" in cls:
                    platform_list.append("Windows")
                if "mac" in cls:
                    platform_list.append("Mac")
                if "linux" in cls:
                    platform_list.append("Linux")
            info["platforms"] = ", ".join(set(platform_list))
        except:
            pass

        # Pricing: check discounted price first, then regular price
        try:
            discount_price = driver.find_element(By.CLASS_NAME, "discount_final_price").text.strip()
            original_price = driver.find_element(By.CLASS_NAME, "discount_original_price").text.strip()
            discount_pct = driver.find_element(By.CLASS_NAME, "discount_pct").text.strip()
            info["current_price"] = original_price
            info["discounted_price"] = discount_price
            info["discount_percent"] = discount_pct
        except:
            try:
                # If no discount, get the normal price
                price_block = driver.find_element(By.CLASS_NAME, "game_purchase_price").text.strip()
                info["current_price"] = price_block
                info["discounted_price"] = ""
                info["discount_percent"] = ""
            except:
                pass

    except Exception as e:
        print("Error getting additional info:", e)

    return info

def scrape_steam_top_games(max_games=2500):
    driver = create_driver()
    games = []

    for page in range(1, 5000):  # Adjust for more pages if needed
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
                    total_reviews = int(match_reviews.group(1).replace(",", "")) if match_reviews else 0
                    positive_pct = int(match_pos.group(1)) if match_pos else 0
                except:
                    total_reviews = 0
                    positive_pct = 0

                game_data = {
                    "game_id": game_id,
                    "name": name,
                    "release_date": release_date_str,
                    "release_year": release_year,
                    "total_reviews": total_reviews,
                    "positive_percent": positive_pct
                }

                # Get detailed info
                additional_info = get_additional_game_info(href, driver)
                game_data.update(additional_info)

                games.append(game_data)

                print(f"Collected: {name} ({release_year})")

                if len(games) >= max_games:
                    break

            except Exception as e:
                print("Error parsing game:", e)

        if len(games) >= max_games:
            break

    driver.quit()

    # Save DataFrame
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(games)

    # Ensure all expected columns are present
    required_columns = ["game_id", "name", "release_date", "release_year", "total_reviews", "positive_percent",
                        "developer", "publisher", "genres", "platforms",
                        "current_price", "discounted_price", "discount_percent"]

    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df.to_csv("data/steam_games.csv", index=False)
    print(f"\n✅ Scraping complete. Total games collected: {len(df)}")
    return df

# Run the scraper
##if __name__ == "__main__":
    ##scrape_steam_top_games()
