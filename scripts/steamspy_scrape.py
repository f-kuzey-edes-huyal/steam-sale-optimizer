import requests
import time
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

def get_steamspy_data(appid, max_retries=3, delay=2):
    """
    Fetch additional game metrics from SteamSpy API.

    Parameters:
    - appid (int): Steam app ID of the game
    - max_retries (int): Number of retry attempts in case of failure
    - delay (int): Delay in seconds between retries

    Returns:
    - dict: SteamSpy data JSON or empty dict if failed
    """
    url = f"https://steamspy.com/api.php?request=appdetails&appid={appid}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Warning: SteamSpy API returned status {response.status_code} for appid {appid}")
        except requests.RequestException as e:
            print(f"Error fetching SteamSpy data for appid {appid}: {e}")

        time.sleep(delay)  # wait before retrying

    print(f"Failed to fetch SteamSpy data for appid {appid} after {max_retries} attempts.")
    return {}

def fetch_and_save_steamspy_data(game_ids, save_path="data/raw/steamspy_data.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steamspy_data_list = []
    for appid in game_ids:
        print(f"Fetching SteamSpy data for AppID: {appid}")
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

    steamspy_df = pd.DataFrame(steamspy_data_list)
    steamspy_df.to_csv(save_path, index=False)
    print(f"SteamSpy data saved to {save_path}")
    return steamspy_df

