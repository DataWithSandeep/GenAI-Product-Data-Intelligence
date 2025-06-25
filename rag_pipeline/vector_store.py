# rag_pipeline/vector_store.py

import faiss
import numpy as np
import pickle
import os

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 similarity
    index.add(embeddings)
    return index

def save_index(index, metadata, index_path='vector_store/index.faiss', meta_path='vector_store/metadata.pkl'):
    os.makedirs('vector_store', exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)

def load_index(index_path='vector_store/index.faiss', meta_path='vector_store/metadata.pkl'):
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def search_index(index, query_embedding, top_k=5):
    D, I = index.search(query_embedding, top_k)
    return I, D  # I: indices, D: distances
