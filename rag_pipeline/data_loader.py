# rag_pipeline/data_loader.py

import pandas as pd
import os

def load_and_merge_data(data_dir='data'):
    # Load CSV files
    item_master = pd.read_csv(os.path.join(data_dir, 'item_master.csv'))
    item_classification = pd.read_csv(os.path.join(data_dir, 'item_classification.csv'))
    item_prices = pd.read_csv(os.path.join(data_dir, 'item_prices.csv'))
    customer_reviews = pd.read_csv(os.path.join(data_dir, 'customer_reviews.csv'))

    # Merge structured data on ITEM_ID
    df = item_master.merge(item_classification, on='ITEM_ID', how='left')
    df = df.merge(item_prices, on='ITEM_ID', how='left')

    # Aggregate reviews: average rating per item
    avg_reviews = customer_reviews.groupby('ITEM_ID')['RATING'].mean().reset_index()
    avg_reviews.rename(columns={'RATING': 'AVG_RATING'}, inplace=True)

    # Merge avg rating
    df = df.merge(avg_reviews, on='ITEM_ID', how='left')

    # Merge review texts into a single blob per item
    review_texts = customer_reviews.groupby('ITEM_ID')['REVIEW_TEXT'].apply(lambda x: " ".join(x)).reset_index()
    review_texts.rename(columns={'REVIEW_TEXT': 'ALL_REVIEWS'}, inplace=True)

    df = df.merge(review_texts, on='ITEM_ID', how='left')

    return df
