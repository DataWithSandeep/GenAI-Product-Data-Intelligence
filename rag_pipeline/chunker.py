# rag_pipeline/chunker.py

import pandas as pd

def chunk_items(df):
    chunks = []

    for _, row in df.iterrows():
        item_id = row['ITEM_ID']
        
        # Build a chunk of information
        text = f"""ITEM NAME: {row['ITEM_NAME']}
DESCRIPTION: {row['ITEM_DESCRIPTION']}
BRAND: {row['BRAND']}
MANUFACTURER: {row['MANUFACTURER']}
UOM: {row['UOM']}
ITEM TYPE: {row['ITEM_TYPE']}
STATUS: {row['STATUS']}

CATEGORY: {row['CATEGORY']}
SUBCATEGORY: {row['SUBCATEGORY']}
CLASS: {row['CLASS']}
HSN CODE: {row['HSN_CODE']}
TAX RATE: {row['TAX_RATE']}
STORAGE CONDITION: {row['STORAGE_CONDITION']}
SHELF LIFE: {row['SHELF_LIFE']}

UNIT PRICE: {row['UNIT_PRICE']}
BULK PRICE: {row['BULK_PRICE']}
MIN ORDER QTY: {row['MIN_ORDER_QTY']}
PRICE VALIDITY: {row['EFFECTIVE_FROM']} to {row['EFFECTIVE_TO']}

AVERAGE RATING: {round(row['AVG_RATING'], 2) if pd.notna(row['AVG_RATING']) else 'N/A'}
CUSTOMER REVIEWS: {row['ALL_REVIEWS'] if pd.notna(row['ALL_REVIEWS']) else 'No reviews available.'}
"""

        chunks.append({
            "item_id": item_id,
            "text": text.strip()
        })

    return chunks
