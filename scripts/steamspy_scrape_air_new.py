import requests
import time
import random
import pandas as pd
import os


def safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_steamspy_data(appid, max_retries=5, delay=2):
    """
    Fetch additional game metrics from SteamSpy API with retries, backoff, and headers.

    Parameters:
    - appid (int): Steam app ID of the game
    - max_retries (int): Number of retry attempts in case of failure
    - delay (int): Base delay in seconds between retries

    Returns:
    - dict: SteamSpy data JSON or empty dict if failed
    """
    url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; MySteamSpyScraper/1.0; +https://example.com)"
    }
    session = requests.Session()
    session.trust_env = False  # Prevent usage of system proxy settings

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=10, headers=headers)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if not data:
                        print(f"Warning: Empty data for appid {appid} on attempt {attempt}")
                    else:
                        return data
                except ValueError as e:
                    print(f"JSON decode error for appid {appid} on attempt {attempt}: {e}")
            elif response.status_code == 429:
                wait = delay * (2 ** attempt)
                print(f"Rate limited (429) for appid {appid}, waiting {wait:.1f}s before retry")
                time.sleep(wait)
                continue
            else:
                print(f"HTTP {response.status_code} for appid {appid} on attempt {attempt}")

        except requests.RequestException as e:
            print(f"Request error for appid {appid} on attempt {attempt}: {e}")

        # Exponential backoff with jitter
        wait = delay * (2 ** attempt) + random.uniform(0, 1)
        print(f"Retrying appid {appid} in {wait:.1f} seconds...")
        time.sleep(wait)

    print(f"Failed to fetch SteamSpy data for appid {appid} after {max_retries} attempts.")
    return {}

def fetch_and_save_steamspy_data(game_ids, save_path="data/steam_api_apache.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steamspy_data_list = []
    for idx, appid in enumerate(game_ids, start=1):
        print(f"[{idx}/{len(game_ids)}] Fetching SteamSpy data for AppID: {appid}")
        steamspy_info = get_steamspy_data(appid)
        steamspy_data = {
            'game_id': appid,
            'owners': steamspy_info.get('owners', ''),
            'average_forever': safe_int(steamspy_info.get('average_forever')),
            'average_2weeks': safe_int(steamspy_info.get('average_2weeks')),
            'median_forever': safe_int(steamspy_info.get('median_forever')),
            'median_2weeks': safe_int(steamspy_info.get('median_2weeks')),
            'players_forever': safe_int(steamspy_info.get('players_forever')),
            'players_2weeks': safe_int(steamspy_info.get('players_2weeks')),
            'price': safe_int(steamspy_info.get('price')),
            'initialprice': safe_int(steamspy_info.get('initialprice')),
            'discount': safe_int(steamspy_info.get('discount')),
            'ccu': safe_int(steamspy_info.get('ccu')),
            'followers': safe_int(steamspy_info.get('followers'))
        }
        steamspy_data_list.append(steamspy_data)

        # Optional: Save periodically every N items to avoid data loss on failure
        if idx % 50 == 0 or idx == len(game_ids):
            steamspy_df = pd.DataFrame(steamspy_data_list)
            steamspy_df.to_csv(save_path, index=False)
            print(f"Progress saved to {save_path} after {idx} items.")

        # Optional: add a small delay between requests to be polite
        time.sleep(1 + random.uniform(0, 0.5))

    steamspy_df = pd.DataFrame(steamspy_data_list)
    steamspy_df.to_csv(save_path, index=False)
    print(f"SteamSpy data saved to {save_path}")
    return steamspy_df
