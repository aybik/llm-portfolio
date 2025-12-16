import os
import time
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

        if end >= len(text):
            break

    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove special characters (optional)
    # text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)

    return text.strip()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for text."""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Rough estimate: ~4 chars per token
        return len(text) // 4
