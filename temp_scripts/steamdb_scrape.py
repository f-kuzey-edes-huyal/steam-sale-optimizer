import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def get_recent_games(pages=5):
    base_url = "https://store.steampowered.com/search/"
    games = []
    three_years_ago = datetime.now() - timedelta(days=365*3)  # 3 years ago

    for page in range(1, pages + 1):
        params = {
            "sort_by": "Released_DESC",
            "page": page,
            "filter": "released",
            "os": "win"
        }
        r = requests.get(base_url, params=params)
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select(".search_result_row")

        for row in rows:
            title = row.select_one(".title").text.strip()
            app_link = row["href"]
            # Extract appid safely
            parts = app_link.rstrip("/").split("/")
            appid = None
            for part in parts:
                if part.isdigit():
                    appid = part
                    break
            if appid is None:
                print(f"Could not find appid in URL: {app_link}")
                continue

            release_str = row.select_one(".search_released").text.strip()
            try:
                release_date = datetime.strptime(release_str, "%b %d, %Y")
                if release_date < three_years_ago:
                    continue  # Skip older games (beyond 3 years)
            except:
                continue

            games.append({
                "game_id": appid,
                "title": title,
                "release_date": release_date.strftime("%Y-%m-%d"),
                "developer": None,  # to be filled later
                "publisher": None,
                "base_price": None
            })

        time.sleep(1)  # polite delay

    return pd.DataFrame(games)


def get_game_details(appid):
    url = f"https://store.steampowered.com/app/{appid}/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    developer = "Unknown"
    publisher = "Unknown"
    base_price = 0.0

    try:
        # Developer(s)
        dev_row = soup.find_all('div', class_='dev_row')
        if dev_row:
            dev_links = dev_row[0].select('a')
            if dev_links:
                developer = ", ".join([a.text.strip() for a in dev_links])

        # Publisher(s) - improved approach
        publisher = "Unknown"
        details_block = soup.select_one('div.details_block')
        if details_block:
            # Look for line that contains Publisher
            text_lines = details_block.get_text(separator="\n").split("\n")
            pub_lines = [line for line in text_lines if "Publisher:" in line]
            if pub_lines:
                # If found, try to get links inside details_block related to Publisher
                pub_links = details_block.select('b:contains("Publisher:") + a, b:contains("Publisher:") + span a')
                if not pub_links:
                    # fallback: find all links and try matching
                    pub_links = []
                    for b_tag in details_block.select('b'):
                        if "Publisher:" in b_tag.text:
                            pub_links = b_tag.parent.select('a')
                            break
                if pub_links:
                    publisher = ", ".join([a.text.strip() for a in pub_links])
                else:
                    # fallback: just get text after "Publisher:"
                    for line in text_lines:
                        if line.startswith("Publisher:"):
                            publisher = line.replace("Publisher:", "").strip()
                            break

        # Price - discounted or normal
        price_elem = soup.select_one('.discount_final_price, .game_purchase_price')
        if price_elem:
            price_text = price_elem.text.strip()
            if "Free" in price_text or "free" in price_text:
                base_price = 0.0
            else:
                # Clean price text (e.g., $19.99, €19.99)
                price_text = price_text.replace("$", "").replace("€", "").replace("£", "").replace(",", "").strip()
                try:
                    base_price = float(price_text)
                except:
                    base_price = 0.0
        else:
            free_label = soup.select_one('.game_area_purchase_game')
            if free_label and "Free to Play" in free_label.text:
                base_price = 0.0

    except Exception as e:
        print(f"Error scraping details for appid {appid}: {e}")

    return developer, publisher, base_price
