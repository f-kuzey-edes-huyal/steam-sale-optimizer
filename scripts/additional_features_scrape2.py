import pandas as pd
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests

# Read game IDs
df_games = pd.read_csv("steam_data/filtered_steam_games.csv")
game_ids = df_games['game_id'].dropna().astype(int).tolist()

# Create headless driver
def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# Improved discount duration detector
def get_discount_duration(driver):
    try:
        elem = driver.find_element(By.CLASS_NAME, 'game_purchase_discount_countdown')
        text = elem.text.lower()

        if "days left" in text:
            return int(text.split()[0])
        elif "hours left" in text:
            return 1
        elif "offer ends" in text:
            for fmt in ['%B %d', '%b %d']:  # e.g. "July 5" or "Jul 5"
                try:
                    end_str = text.split("offer ends")[-1].strip()
                    end_date = datetime.strptime(end_str, fmt).replace(year=datetime.now().year)
                    return (end_date.date() - datetime.today().date()).days
                except ValueError:
                    continue
        return 7  # Fallback default if structure is unclear
    except:
        return 0  # No discount info found

# Check if seasonal sale
def is_seasonal_sale():
    today = datetime.today().date()
    seasonal_sales = [
        (datetime(2025, 3, 13).date(), datetime(2025, 3, 20).date()),  # Spring Sale
        (datetime(2025, 6, 26).date(), datetime(2025, 7, 10).date()),  # Summer Sale
        (datetime(2025, 9, 29).date(), datetime(2025, 10, 6).date()),  # Autumn/Halloween Warmup
        (datetime(2025, 12, 18).date(), datetime(2026, 1, 5).date()),  # Winter Sale
    ]
    return any(start <= today <= end for start, end in seasonal_sales)

# Robust game age parser
def get_game_age(driver):
    date_selectors = [
        ".release_date .date",
        ".details_block > b + br + br",
    ]
    for selector in date_selectors:
        try:
            elem = driver.find_element(By.CSS_SELECTOR, selector)
            text = elem.text.strip()
            for fmt in ['%b %d, %Y', '%d %b, %Y', '%Y']:
                try:
                    release_date = datetime.strptime(text, fmt).date()
                    return (datetime.today().date() - release_date).days
                except ValueError:
                    continue
        except:
            continue
    return None

# Get total reviews from API
def get_total_reviews(app_id):
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=all&num_per_page=0"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("query_summary", {}).get("total_reviews", 0)
    except:
        pass
    return None

# Main scraping logic
driver = create_driver()
season_flag = is_seasonal_sale()
data = []

for app_id in game_ids:
    print(f"Processing {app_id}")
    try:
        url = f"https://store.steampowered.com/app/{app_id}/"
        driver.get(url)
        time.sleep(2)

        row = {
            "game_id": app_id,
            "discount_duration": get_discount_duration(driver),
            "is_seasonal_sale": season_flag,
            "game_age_days": get_game_age(driver),
            "total_reviews": get_total_reviews(app_id),
        }

        data.append(row)
    except Exception as e:
        print(f"Error processing {app_id}: {e}")
        continue

driver.quit()

# Save to CSV
df_output = pd.DataFrame(data)
df_output.to_csv("steam_data/additional_steam_features2.csv", index=False)
print(" Saved to additional_steam_features2.csv")
