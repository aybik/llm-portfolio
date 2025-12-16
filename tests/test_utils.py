import sys
sys.path.append('..')

from utils.prompts import PromptTemplate
from utils.metrics import RAGMetrics
from utils.text_processing import chunk_text, count_tokens

def test_prompt_formatting():
    prompt = PromptTemplate.format(
        PromptTemplate.QA_TEMPLATE,
        context="Python is a programming language.",
        question="What is Python?"
    )
    assert "Python is a programming language" in prompt
    assert "What is Python?" in prompt
    print("✓ Prompt formatting works")

def test_metrics():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc1", "doc3", "doc4"]

    precision = RAGMetrics.retrieval_precision(retrieved, relevant)
    recall = RAGMetrics.retrieval_recall(retrieved, relevant)

    assert precision == 2/3  # 2 relevant out of 3 retrieved
    assert recall == 2/3  # 2 relevant retrieved out of 3 total relevant
    print("✓ Metrics calculation works")

def test_text_processing():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    print(f"✓ Text chunking works - created {len(chunks)} chunks")

if __name__ == "__main__":
    test_prompt_formatting()
    test_metrics()
    test_text_processing()
    print("\n✅ All utils tests passed!")
