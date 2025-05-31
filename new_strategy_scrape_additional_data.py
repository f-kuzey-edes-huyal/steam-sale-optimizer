import pandas as pd
import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def create_driver():
    options = Options()
    options.add_argument("--headless")  # Run browser in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def get_review_data(app_id):
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=all&num_per_page=0"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json().get("query_summary", {})
    total_reviews = data.get("total_reviews", 0)
    total_positive = data.get("total_positive", 0)
    total_negative = data.get("total_negative", 0)
    review_score_desc = data.get("review_score_desc", "")
    return {
        "total_reviews": total_reviews,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "review_score_desc": review_score_desc
    }

def get_engagement_metrics(app_id, driver):
    url = f"https://store.steampowered.com/app/{app_id}/"
    driver.get(url)
    time.sleep(3)  # Wait for the page to load

    # Get number of followers
    try:
        followers_elem = driver.find_element(By.CSS_SELECTOR, ".apphub_NumFollowers")
        followers = followers_elem.text.strip()
    except:
        followers = None

    # Get user-generated tags
    try:
        tags_elems = driver.find_elements(By.CSS_SELECTOR, ".app_tag")
        tags = [tag.text.strip() for tag in tags_elems if tag.text.strip()]
    except:
        tags = []

    return {
        "followers": followers,
        "tags": tags
    }

def scrape_steam_data(game_ids):
    driver = create_driver()
    data_list = []

    for app_id in game_ids:
        print(f"Processing App ID: {app_id}")
        review_data = get_review_data(app_id)
        engagement_data = get_engagement_metrics(app_id, driver)

        data = {
            "game_id": app_id,
            "total_reviews": review_data.get("total_reviews") if review_data else None,
            "total_positive": review_data.get("total_positive") if review_data else None,
            "total_negative": review_data.get("total_negative") if review_data else None,
            "review_score_desc": review_data.get("review_score_desc") if review_data else None,
            "followers": engagement_data.get("followers"),
            "tags": engagement_data.get("tags")
        }
        data_list.append(data)

    driver.quit()
    return pd.DataFrame(data_list)

if __name__ == "__main__":
    # Read game IDs from the CSV file
    df_games = pd.read_csv("filtered_steam_games.csv")
    game_ids = df_games["game_id"].tolist()

    # Scrape data for each game
    df_data = scrape_steam_data(game_ids)

    # Save the data to a new CSV file
    df_data.to_csv("steam_game_data.csv", index=False)
    print("Data scraping complete!")
