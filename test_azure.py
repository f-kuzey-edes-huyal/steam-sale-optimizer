import requests

url = "https://kuzey-ml-app.azurewebsites.net/predict"

test_data = {
    "game_id": 123,
    "name": "Test Game",
    "release_date": "2023-01-01",
    "total_reviews": 1000,
    "positive_percent": 75,
    "genres": "Action, Adventure",
    "tags": "Multiplayer;Co-op",
    "current_price": "$20.00",
    "discounted_price": "$10.00",
    "owners": "50000..100000",
    "days_after_publish": 365,
    "review": "Great game, loved it!",
    "owner_min": 50000,
    "owner_max": 100000,
    "owners_log_mean": 11.5
}

response = requests.post(url, json=test_data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
