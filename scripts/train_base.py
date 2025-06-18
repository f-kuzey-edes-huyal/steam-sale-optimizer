import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import joblib  # <-- added for saving

# Load data
df = pd.read_csv("data/combined3.csv")

# Drop rows with missing key values
df = df.dropna(subset=[
    'total_reviews', 'positive_percent', 'genres', 'tags',
    'current_price', 'discounted_price', 'owners_log_mean', 'days_after_publish'
])

# Convert prices to float
def parse_price(val):
    try:
        return float(str(val).replace('$', '').replace('USD', '').strip())
    except:
        return np.nan

df['current_price'] = df['current_price'].apply(parse_price)
df['discounted_price'] = df['discounted_price'].apply(parse_price)
df = df.dropna(subset=['current_price', 'discounted_price'])

# Add discount percentage
df['discount_pct'] = 1 - (df['discounted_price'] / df['current_price'])

# Clean and encode genres and tags
df['genres'] = df['genres'].fillna('').apply(lambda x: [g.strip() for g in str(x).split(',')])
df['tags'] = df['tags'].fillna('').apply(lambda x: [t.strip() for t in str(x).split(';')])

mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()

genres_encoded = pd.DataFrame(mlb_genres.fit_transform(df['genres']), columns=mlb_genres.classes_)
tags_encoded = pd.DataFrame(mlb_tags.fit_transform(df['tags']), columns=mlb_tags.classes_)

# Concatenate encoded features
df_enc = pd.concat([df.reset_index(drop=True), genres_encoded, tags_encoded], axis=1)

# Features and target
features = [
    'total_reviews', 'positive_percent', 'current_price',
    'owners_log_mean', 'days_after_publish'
] + list(genres_encoded.columns) + list(tags_encoded.columns)

X = df_enc[features]
y = df_enc['discount_pct']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
num_features = ['total_reviews', 'positive_percent', 'current_price', 'owners_log_mean', 'days_after_publish']
num_transformer = Pipeline([('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features)
], remainder='passthrough')

# Full pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5]
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Best Params: {grid_search.best_params_}")
print(f"RMSE on Test Set: {rmse:.4f}")

# Save final model and preprocessor
joblib.dump(best_model, 'models/discount_model_pipeline.pkl')
joblib.dump(mlb_genres, 'models/mlb_genres.pkl')
joblib.dump(mlb_tags, 'models/mlb_tags.pkl')
