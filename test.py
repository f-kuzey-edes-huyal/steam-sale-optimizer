# test_predict.py

import requests

data = {
    "total_reviews": 1000,
    "positive_percent": 95.0,
    "current_price": 20.0,
    "discounted_price": 10.0,
    "owners_log_mean": 13.5,
    "days_after_publish": 100,
    "genres": ["Action", "Indie"],
    "tags": ["Multiplayer", "Co-op"],
    "review": "Great game with fantastic gameplay and graphics!"
}

response = requests.post("http://localhost:8000/predict", json=data)

print("Status Code:", response.status_code)
print("Response:", response.json())
