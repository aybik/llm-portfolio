from typing import List, Dict
class RAGMetrics:
    """Metrics for evaluating RAG systems."""

    @staticmethod
    def retrieval_precision(
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """Calculate precision: relevant_retrieved / total_retrieved."""
        if not retrieved_docs:
            return 0.0

        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
        return len(relevant_retrieved) / len(retrieved_docs)

    @staticmethod
    def retrieval_recall(
        retrieved_docs: List[str],
        relevant_docs: List[str]
    ) -> float:
        """Calculate recall: relevant_retrieved / total_relevant."""
        if not relevant_docs:
            return 0.0

        relevant_retrieved = set(retrieved_docs) & set(relevant_docs)
        return len(relevant_retrieved) / len(relevant_docs)

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def mean_reciprocal_rank(rankings: List[List[str]], relevant: List[str]) -> float:
        """Calculate MRR."""
        reciprocal_ranks = []

        for ranking in rankings:
            for i, doc in enumerate(ranking, 1):
                if doc in relevant:
                    reciprocal_ranks.append(1 / i)
                    break
            else:
                reciprocal_ranks.append(0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)


class GenerationMetrics:
    """Metrics for evaluating text generation."""

    @staticmethod
    def rouge_score(generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores (simplified)."""
        try:
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(generated, reference)[0]
            return {
                "rouge-1": scores["rouge-1"]["f"],
                "rouge-2": scores["rouge-2"]["f"],
                "rouge-l": scores["rouge-l"]["f"]
            }
        except ImportError:
            print("Install rouge package: pip install rouge")
            return {}

    @staticmethod
    def bleu_score(generated: str, reference: str) -> float:
        """Calculate BLEU score (simplified)."""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            reference_tokens = [reference.split()]
            generated_tokens = generated.split()
            return sentence_bleu(reference_tokens, generated_tokens)
        except ImportError:
            print("Install nltk: pip install nltk")
            return 0.0
