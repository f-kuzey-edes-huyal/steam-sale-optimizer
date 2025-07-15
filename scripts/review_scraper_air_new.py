import os
import requests
import time
from textblob import TextBlob


def get_reviews(appid, num_reviews=100):
    # Clear proxy env variables so requests won't try to use the invalid proxy
    for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ.pop(var, None)

    reviews = []
    cursor = '*'
    previous_cursor = None
    count = 0

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; SteamScraper/1.0)'
    }

    while count < num_reviews:
        url = (
            f"https://store.steampowered.com/appreviews/{appid}"
            f"?json=1&filter=recent&language=all&cursor={cursor}&num_per_page=100"
        )

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.ProxyError as e:
            print("Request failed due to proxy error:", e)
            break
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            break

        data = response.json()
        current_reviews = data.get('reviews', [])
        if not current_reviews:
            print("No more reviews found.")
            break

        for review in current_reviews:
            review_text = review.get('review', '')
            sentiment = TextBlob(review_text).sentiment.polarity
            reviews.append({
                'review': review_text,
                'sentiment': sentiment,
                'timestamp_created': review.get('timestamp_created'),
                'voted_up': review.get('voted_up')
            })
            count += 1
            if count >= num_reviews:
                break

        previous_cursor = cursor
        cursor = data.get('cursor')

        if cursor == previous_cursor or cursor is None:
            print("Same or null cursor received. Breaking to avoid infinite loop.")
            break

        time.sleep(1)

    return reviews
