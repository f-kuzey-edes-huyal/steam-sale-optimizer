# transformers.py
import numpy as np

class CompetitorPricingTransformer:
    def __init__(self):
        self.tag_price_map = {}

    def fit(self, df):
        tag_prices = {}
        for tags, price in zip(df['tags'], df['current_price']):
            for tag in tags:
                tag_prices.setdefault(tag, []).append(price)
        self.tag_price_map = {tag: np.median(prices) for tag, prices in tag_prices.items()}
        return self

    def transform(self, df):
        def competitor_price_for_tags(tags):
            prices = [self.tag_price_map.get(tag, np.nan) for tag in tags]
            prices = [p for p in prices if not np.isnan(p)]
            if prices:
                return np.mean(prices)
            else:
                return np.nan

        df['competitor_pricing'] = df['tags'].apply(competitor_price_for_tags)
        median_price = df['current_price'].median()
        df['competitor_pricing'] = df['competitor_pricing'].fillna(median_price)
        return df
