from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

def scrape_pricing_for_games(df_games):
    driver = create_driver()
    pricing_list = []

    for idx, row in df_games.iterrows():
        game_id = row["game_id"]
        url = f"https://store.steampowered.com/app/{game_id}/"
        driver.get(url)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.game_area_purchase_section"))
            )

            discount_price = None
            original_price = None
            discount_pct = 0

            if driver.find_elements(By.CSS_SELECTOR, "div.discount_final_price"):
                discount_price = driver.find_element(By.CSS_SELECTOR, "div.discount_final_price").text.strip()
                original_price = driver.find_element(By.CSS_SELECTOR, "div.discount_original_price").text.strip()
                op = float(re.sub(r"[^\d.]", "", original_price))
                dp = float(re.sub(r"[^\d.]", "", discount_price))
                discount_pct = round((1 - dp / op) * 100, 2)

            elif driver.find_elements(By.CSS_SELECTOR, "div.game_purchase_price.price"):
                price_elem = driver.find_element(By.CSS_SELECTOR, "div.game_purchase_price.price")
                original_price = price_elem.text.strip()
                discount_price = original_price
                discount_pct = 0
            else:
                raise ValueError("No pricing info found.")

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

        time.sleep(random.uniform(2.5, 4.0))  # Optional delay

    driver.quit()
    return pd.DataFrame(pricing_list)

if __name__ == "__main__":
    #df_games = pd.read_csv("steam_data/steam_2020_to_2025_games.csv")
    df_games = pd.read_csv("steam_data/all_steam_games.csv")

    df_pricing = scrape_pricing_for_games(df_games)
    print(df_pricing.head())
    df_pricing.to_csv("steam_data/steam_pricing.csv", index=False)
    print("Scraping complete!")
