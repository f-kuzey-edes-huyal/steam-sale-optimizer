from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from dateutil import parser
import time
import pandas as pd
import re
import random
import os


def create_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-proxy-server")
    options.add_argument("--proxy-server='direct://'")
    options.add_argument("--proxy-bypass-list=*")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36")

    service = Service("/usr/local/bin/chromedriver")
    options.binary_location = "/usr/bin/google-chrome"
    return webdriver.Chrome(service=service, options=options)


def parse_release_year(release_str):
    try:
        dt = parser.parse(release_str, fuzzy=True)
        return dt.year
    except Exception:
        return None


def get_additional_game_info(game_url, driver):
    info = {
        "developer": "",
        "publisher": "",
        "genres": "",
        "tags": "",
        "platforms": "",
        "current_price": "",
        "discounted_price": "",
        "discount_percent": ""
    }

    try:
        driver.get(game_url)
        time.sleep(random.uniform(2, 4))

        # Developer & Publisher
        try:
            labels = driver.find_elements(By.CSS_SELECTOR, ".details_block .dev_row .subtitle")
            values = driver.find_elements(By.CSS_SELECTOR, ".details_block .dev_row .summary")
            for label, value in zip(labels, values):
                lbl_text = label.text.strip().lower()
                val_text = value.text.strip()
                if "developer" in lbl_text:
                    info["developer"] = val_text
                elif "publisher" in lbl_text:
                    info["publisher"] = val_text
        except:
            pass

        # Genres & Tags
        try:
            genre_links = driver.find_elements(By.CSS_SELECTOR, ".details_block a[href*='genre']")
            genres = [g.text.strip() for g in genre_links if g.text.strip()]
            info["genres"] = ", ".join(genres)
            info["tags"] = ";".join(genres)
        except:
            pass

        # Platforms
        try:
            platform_imgs = driver.find_elements(By.CSS_SELECTOR, ".game_area_purchase_platform .platform_img")
            platforms = []
            for img in platform_imgs:
                cls = img.get_attribute("class")
                if "win" in cls:
                    platforms.append("Windows")
                if "mac" in cls:
                    platforms.append("Mac")
                if "linux" in cls:
                    platforms.append("Linux")
            info["platforms"] = ", ".join(sorted(set(platforms)))
        except:
            pass

        # Pricing
        try:
            discount_price = driver.find_element(By.CLASS_NAME, "discount_final_price").text.strip()
            original_price = driver.find_element(By.CLASS_NAME, "discount_original_price").text.strip()
            discount_pct = driver.find_element(By.CLASS_NAME, "discount_pct").text.strip()
            info["current_price"] = original_price
            info["discounted_price"] = discount_price
            info["discount_percent"] = discount_pct
        except:
            try:
                price_elem = driver.find_element(By.CSS_SELECTOR, ".game_purchase_price, .price")
                price = price_elem.text.strip()
                info["current_price"] = price
                info["discounted_price"] = price
                info["discount_percent"] = ""
            except:
                info["current_price"] = "Free To Play"
                info["discounted_price"] = "Free To Play"
                info["discount_percent"] = ""

    except Exception as e:
        print(f"Error getting additional info for {game_url}: {e}")

    return info


def scrape_steam_top_games(max_games=10):
    driver_main = None
    driver_detail = None

    try:
        driver_main = create_driver()
        driver_detail = create_driver()
        games = []

        for page in range(1, 10000):
            print(f"Scraping page {page}...")
            url = f"https://store.steampowered.com/search/?filter=topsellers&ignore_preferences=1&page={page}"
            driver_main.get(url)
            time.sleep(random.uniform(2, 4))

            rows = driver_main.find_elements(By.CSS_SELECTOR, "a.search_result_row")

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

                    additional_info = get_additional_game_info(href, driver_detail)
                    game_data.update(additional_info)

                    games.append(game_data)
                    print(f"Collected: {name} ({release_year})")

                    if len(games) >= max_games:
                        break

                except Exception as e:
                    print(f"Error parsing game: {e}")

            if len(games) >= max_games:
                break

        df = pd.DataFrame(games)

        required_columns = [
            "game_id", "name", "release_date", "release_year",
            "total_reviews", "positive_percent",
            "developer", "publisher", "genres", "tags", "platforms",
            "current_price", "discounted_price", "discount_percent"
        ]
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/steamdata_apache.csv", index=False)
        print(f"Scraping complete. Total games collected: {len(df)}")
        return df

    finally:
        if driver_main:
            driver_main.quit()
        if driver_detail:
            driver_detail.quit()
