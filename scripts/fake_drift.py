import pandas as pd
import numpy as np
import os

# Load the reference dataset
df = pd.read_csv("data/combined4.csv")

drifted_batches = []

for day in range(6):  # days 0,1,2,3,4,5
    # Filter rows with days_after_publish > 5 for drift candidates
    df_drift = df[df['days_after_publish'] > 5].copy()

    # Duplicate if too few rows
    if len(df_drift) < 5:
        df_drift = pd.concat([df_drift]*5, ignore_index=True)

    # Sample 5 rows
    df_sample = df_drift.sample(n=5, random_state=42 + day).copy()  # change seed per day for variability
    df_sample['days_after_publish'] = day

    # Add drift:
    df_sample['current_price'] = df_sample['current_price'].astype(str).str.replace('$', '').str.replace('USD', '').str.strip()
    df_sample['current_price'] = df_sample['current_price'].astype(float) * (1 + 0.1 * day)  # increase price progressively
    df_sample['current_price'] = df_sample['current_price'].apply(lambda x: f"${x:.2f}")

    df_sample['tags'] = df_sample['tags'].astype(str).apply(lambda x: x + ";Experimental")

    df_sample['positive_percent'] = df_sample['positive_percent'].apply(lambda x: max(x - np.random.randint(2, 5), 40))

    df_sample['review'] = df_sample['review'].astype(str) + " Some recent issues make it less enjoyable."

    drifted_batches.append(df_sample)

df_drift_all_days = pd.concat(drifted_batches, ignore_index=True)

os.makedirs("data", exist_ok=True)
df_drift_all_days.to_csv("data/drifted_data.csv", index=False)

print("Drifted data for days 0 to 5 saved to data/drifted_data.csv")
