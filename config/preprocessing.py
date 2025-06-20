from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

NUMERIC_FEATURES = ['total_reviews', 'positive_percent', 'current_price', 'owners_log_mean', 'days_after_publish']

def get_preprocessor():
    num_transformer = Pipeline([('scaler', StandardScaler())])
    preprocessor = ColumnTransformer([('num', num_transformer, NUMERIC_FEATURES)], remainder='passthrough')
    return preprocessor
