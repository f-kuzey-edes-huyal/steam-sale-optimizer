import requests
import time
from textblob import TextBlob


def get_reviews(appid, num_reviews=100):
    reviews = []
    cursor = '*'
    previous_cursor = None
    count = 0

    while count < num_reviews:
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent&language=all&cursor={cursor}&num_per_page=100"
        response = requests.get(url)
        if response.status_code != 200:
            print("❌ Request failed")
            break
        data = response.json()

        current_reviews = data.get('reviews', [])
        if not current_reviews:
            print("ℹ️ No more reviews found.")
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

        # 🚨 Check for stuck cursor
        if cursor == previous_cursor:
            print("⚠️ Same cursor received again. Breaking to avoid infinite loop.")
            break

        time.sleep(1)

    return reviews
