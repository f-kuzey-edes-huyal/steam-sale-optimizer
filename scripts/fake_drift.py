import pandas as pd
import numpy as np
import os

file_path = r'data/combined4.csv'

# Step 1: Try loading CSV with automatic delimiter detection, fallback to semicolon if fails
try:
    df = pd.read_csv(file_path, sep=None, engine='python', parse_dates=['release_date'])
    print("Loaded data with auto-detected delimiter.")
except Exception as e:
    print(f"Auto delimiter failed: {e}")
    try:
        df = pd.read_csv(file_path, sep=';', parse_dates=['release_date'])
        print("Loaded data with semicolon separator.")
    except Exception as e2:
        print(f"Failed to load CSV with semicolon separator too: {e2}")
        raise

# Step 2: Clean 'current_price' and 'discounted_price' columns by removing currency symbols and converting to float
def clean_price(p):
    try:
        if isinstance(p, str):
            return float(p.replace('$', '').replace(' USD', '').replace(',', '').strip())
        return float(p)
    except Exception:
        return np.nan

df['current_price'] = df['current_price'].apply(clean_price)
df['discounted_price'] = df['discounted_price'].apply(clean_price)

# Step 3: Add 'days_after_publish' column based on current date minus release_date if not present
if 'days_after_publish' not in df.columns:
    today = pd.Timestamp.today()
    # Calculate days after publish (difference in days between today and release_date)
    df['days_after_publish'] = (today - df['release_date']).dt.days
    # Fill negative or NaN values with 0
    df['days_after_publish'] = df['days_after_publish'].apply(lambda x: x if x >= 0 else 0)
    df['days_after_publish'] = df['days_after_publish'].fillna(0).astype(int)

# Step 4: Create extended dataset of 300 rows, simulate data drift for demonstration
target_rows = 300
original_len = len(df)
np.random.seed(42)  # Fix random seed for reproducibility

extended_rows = []
for i in range(target_rows):
    base = df.iloc[i % original_len].copy()

    # Increase days_after_publish by 5 days per row
    base['days_after_publish'] = int(base.get('days_after_publish', 0)) + i * 5

    # Shift release_date forward by i*5 days if date is valid
    if pd.notnull(base.get('release_date')):
        base['release_date'] += pd.Timedelta(days=i * 5)

    # Simulate growth and noise in total_reviews
    base['total_reviews'] = int(
        base.get('total_reviews', 0) * (1 + 0.01 * (i // original_len))
        + np.random.randint(-5, 10)
    )

    # Add small noise to positive_percent, keep it between 50 and 100
    base['positive_percent'] = min(
        100, max(50, base.get('positive_percent', 75) + np.random.normal(0, 0.5))
    )

    # Apply small noise drift to prices, keep non-negative
    base['current_price'] = max(0, base.get('current_price', 0) * (1 + np.random.normal(0, 0.01)))
    base['discounted_price'] = max(0, base.get('discounted_price', 0) * (1 + np.random.normal(0, 0.01)))

    # Drift owner_min/max with noise, ensure owner_max >= owner_min
    base['owner_min'] = max(0, base.get('owner_min', 0) * (1 + np.random.normal(0, 0.01)))
    base['owner_max'] = max(base['owner_min'], base.get('owner_max', base['owner_min']) * (1 + np.random.normal(0, 0.01)))

    # Calculate geometric mean for owners_log_mean
    base['owners_log_mean'] = np.sqrt(base['owner_min'] * base['owner_max'])

    # Add update tag to review text, handle missing reviews
    review_text = base.get('review', '')
    if pd.isna(review_text):
        review_text = ''
    base['review'] = review_text + f" [Update {i}]"

    extended_rows.append(base)

# Convert list of rows to DataFrame and reset index
df_extended = pd.DataFrame(extended_rows).reset_index(drop=True)

print("Sample of drifted data:")
print(df_extended.head(3))

# Save extended data to CSV file, create directory if needed
os.makedirs("data", exist_ok=True)
output_path = 'data/drifted_data.csv'
df_extended.to_csv(output_path, index=False)
print(f"Drifted dataset saved to '{output_path}'")
