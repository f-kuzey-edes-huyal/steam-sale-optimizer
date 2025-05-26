from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time

def create_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def parse_release_date(date_str):
    date_formats = ["%b %d, %Y", "%d %b, %Y", "%B %d, %Y", "%d %B, %Y"]
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def get_recent_games(pages=5):
    base_url = "https://store.steampowered.com/search/"
    games = []
    three_years_ago = datetime.now() - timedelta(days=365 * 3)
    driver = create_driver()

    for page in range(1, pages + 1):
        url = f"{base_url}?sort_by=Released_DESC&page={page}&filter=released&os=win"
        print(f"[INFO] Fetching page {page}: {url}")
        driver.get(url)
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        rows = soup.select(".search_result_row")

        for row in rows:
            title_tag = row.select_one(".title")
            release_tag = row.select_one(".search_released")

            if not title_tag or not release_tag:
                continue

            title = title_tag.text.strip()
            app_link = row["href"]
            release_str = release_tag.text.strip()

            if "Coming soon" in release_str or not release_str:
                print(f"[SKIP] Coming soon or empty: '{release_str}'")
                continue

            release_date = parse_release_date(release_str)
            if not release_date:
                print(f"[SKIP] Could not parse release date: '{release_str}'")
                continue

            if release_date < three_years_ago:
                continue

            appid = None
            for part in app_link.rstrip("/").split("/"):
                if part.isdigit():
                    appid = part
                    break

            if not appid:
                print(f"[SKIP] No appid found in {app_link}")
                continue

            games.append({
                "game_id": appid,
                "title": title,
                "release_date": release_date.strftime("%Y-%m-%d"),
                "developer": None,
                "publisher": None,
                "base_price": None
            })

        print(f"[INFO] Page {page} -> Collected {len(games)} games so far.")
        time.sleep(1)

    driver.quit()
    return pd.DataFrame(games)

