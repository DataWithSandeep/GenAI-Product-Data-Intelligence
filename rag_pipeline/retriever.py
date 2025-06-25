# rag_pipeline/retriever.py

from rag_pipeline.vector_store import load_index, search_index
from rag_pipeline.embedder import get_embedder
from rag_pipeline.utils.logger import get_logger
logger = get_logger(__name__)

import numpy as np

class Retriever:
    def __init__(self, index_path='vector_store/index.faiss', meta_path='vector_store/metadata.pkl'):
        self.index, self.metadata = load_index(index_path, meta_path)
        self.model = get_embedder()

    def retrieve(self, query, top_k=5):
        logger.info(f"📥 Received query: {query}")  # ✅ Log the query
        query_vec = self.model.encode([query], convert_to_numpy=True)
        I, D = search_index(self.index, query_vec, top_k)

        results = []
        for i, dist in zip(I[0], D[0]):
            doc = self.metadata[i]
            results.append({
                "item_id": doc["item_id"],
                "text": doc["text"],
                "score": float(dist)
            })
        logger.info(f"🔍 Retrieved top {len(results)} chunks")  # ✅ Log retrieval count
        return results
