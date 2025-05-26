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

def get_game_details(driver, app_link):
    try:
        driver.get(app_link)
        time.sleep(1.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        developer_tag = soup.find("div", class_="dev_row").find("a")
        developer = developer_tag.text.strip() if developer_tag else None

        publisher = None
        publisher_row = soup.find_all("div", class_="dev_row")
        for row in publisher_row:
            if "Publisher" in row.text:
                pub_tag = row.find("a")
                publisher = pub_tag.text.strip() if pub_tag else None
                break

        price_tag = soup.select_one(".game_purchase_price, .discount_final_price")
        price = price_tag.text.strip() if price_tag else None

        return developer, publisher, price
    except Exception as e:
        print(f"[WARN] Failed to get details for {app_link}: {e}")
        return None, None, None

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

            developer, publisher, base_price = get_game_details(driver, app_link)

            games.append({
                "game_id": appid,
                "title": title,
                "release_date": release_date.strftime("%Y-%m-%d"),
                "developer": developer,
                "publisher": publisher,
                "base_price": base_price
            })

        print(f"[INFO] Page {page} -> Collected {len(games)} games so far.")
        time.sleep(1)

    driver.quit()
    return pd.DataFrame(games)
