from typing import List, Dict

class PromptTemplate:
    """Reusable prompt templates."""

    # Question Answering
    QA_TEMPLATE = """Use the following context to answer the question.
If you don't know the answer, say so - don't make up information.

Context:
{context}

Question: {question}

Answer:"""

    # Summarization
    SUMMARIZE_TEMPLATE = """Summarize the following text concisely:

Text:
{text}

Summary:"""

    # Classification
    CLASSIFY_TEMPLATE = """Classify the following text into one of these categories: {categories}

Text: {text}

Category:"""

    # Data Extraction
    EXTRACT_TEMPLATE = """Extract {fields} from the following text in JSON format:

Text:
{text}

JSON:"""

    # Chain of Thought
    COT_TEMPLATE = """Let's solve this step by step:

Problem: {problem}

Step-by-step solution:"""

    # Few-Shot Learning
    FEW_SHOT_TEMPLATE = """Here are some examples:

{examples}

Now complete this:
{input}

Output:"""

    @staticmethod
    def format(template: str, **kwargs) -> str:
        """Format template with variables."""
        return template.format(**kwargs)

    @staticmethod
    def create_few_shot_examples(examples: List[Dict[str, str]]) -> str:
        """Create few-shot examples string."""
        formatted = []
        for ex in examples:
            formatted.append(f"Input: {ex['input']}\nOutput: {ex['output']}")
        return "\n\n".join(formatted)
