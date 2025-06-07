import requests
import time
from textblob import TextBlob

def get_reviews(appid, num_reviews=100):
    reviews = []
    cursor = '*'
    count = 0

    while count < num_reviews:
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&filter=recent&language=all&cursor={cursor}&num_per_page=100"
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        for review in data.get('reviews', []):
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
        cursor = data.get('cursor')
        if not cursor:
            break
        time.sleep(1)  # To respect rate limits
    return reviews
