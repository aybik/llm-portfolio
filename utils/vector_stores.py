import time
from typing import List, Dict, Callable
import numpy as np


class VectorStore:
    """Abstract base for vector stores."""

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        raise NotImplementedError

    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        raise NotImplementedError

    def delete(self, ids: List[str]):
        raise NotImplementedError


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return dot_product / (norm1 * norm2)


def batch_embeddings(
    texts: List[str],
    embedding_fn: Callable,
    batch_size: int = 100
) -> List[List[float]]:
    """Batch embed texts to avoid rate limits."""
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_fn(batch)
        embeddings.extend(batch_embeddings)

        # Rate limiting
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return embeddings
