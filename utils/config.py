import os
from typing import List, Dict, Any, Optional, Callable


class Config:
    """Configuration management."""

    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = config_dict or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set config value."""
        self._config[key] = value

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("MODEL", "gpt-4"),
            "temperature": float(os.getenv("TEMPERATURE", "0.0")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "2000")),
        }
        return cls(config)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Using embeddings
    print("Testing embeddings...")
    embedder = get_embedding_provider("local")
    texts = ["Hello world", "Machine learning is fun"]
    embeddings = embedder.embed_documents(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Example: Using prompts
    print("\nTesting prompt templates...")
    prompt = PromptTemplate.format(
        PromptTemplate.QA_TEMPLATE,
        context="The sky is blue because of Rayleigh scattering.",
        question="Why is the sky blue?"
    )
    print(prompt)

    # Example: Calculating metrics
    print("\nTesting metrics...")
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3", "doc4"]
    precision = RAGMetrics.retrieval_precision(retrieved, relevant)
    recall = RAGMetrics.retrieval_recall(retrieved, relevant)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Example: Cost estimation
    print("\nTesting cost estimation...")
    cost = CostEstimator.estimate_cost("gpt-4", input_tokens=1000, output_tokens=500)
    print(f"Estimated cost: ${cost:.4f}")
