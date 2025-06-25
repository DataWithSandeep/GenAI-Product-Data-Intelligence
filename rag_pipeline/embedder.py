# rag_pipeline/embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np

def get_embedder(model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model

def embed_chunks(model, chunks):
    texts = [chunk["text"] for chunk in chunks]
    metadata = [{"item_id": chunk["item_id"], "text": chunk["text"]} for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, metadata
