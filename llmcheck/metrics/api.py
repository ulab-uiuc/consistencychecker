from typing import List, Optional

import litellm
import numpy as np
from transformers import AutoTokenizer

from llmcheck.metrics.base import BaseSimilarityMetric


class APIBasedSimilarity(BaseSimilarityMetric):
    name: str = "embedding similarity"
    # Common embedding models and their token limits

    def __init__(self,
                 model: str,
                 api_base: str = '',
                 max_tokens: Optional[int] = None):
        self.model = model
        self.api_base = api_base
        if api_base:
            print(f"[INFO] Overriding API embedding model with set value: {model}")

        # Set token limit based on model or user-specified limit
        self.max_tokens = 8191

        # Initialize tokenizer for length checking
        # Using gpt2 tokenizer as a reasonable default for token counting
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within model's token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            print(f"[WARNING] Text exceeded token limit ({len(tokens)} > {self.max_tokens}). Truncating...")
            truncated_tokens = tokens[:self.max_tokens]
            text_truncated: str = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            return text_truncated
        return text

    def _get_embedding(self, text: str) -> np.ndarray[float, np.dtype[np.float64]]:
        """Fetch the embedding vector for a given text using the specified model."""
        # Truncate text if necessary
        truncated_text = self._truncate_text(text)

        try:
            if self.api_base:
                response = litellm.embedding(
                    model=self.model,
                    input=truncated_text,
                    api_base=self.api_base
                )
            else:
                response = litellm.embedding(
                    model=self.model,
                    input=truncated_text,
                )

            embedding = response['data'][0]['embedding']
            return np.array(embedding)
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to retrieve embedding: {e}")
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the cosine similarity between embeddings of two texts.
        Returns tuple of (similarity_score, was_truncated).
        """

        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)

        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Zero vector encountered in embeddings.")

        similarity = dot_product / (norm1 * norm2)
        similarity_float = similarity.item()
        assert isinstance(similarity_float, float)

        return similarity_float

    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        """
        Calculate similarities for multiple pairs of texts.
        Returns list of tuples (similarity_score, was_truncated).
        """
        if len(texts1) != len(texts2):
            raise ValueError("Input lists must have the same length.")

        results = []
        for t1, t2 in zip(texts1, texts2):
            results.append(self.calculate_similarity(t1, t2))
        return results

# Example Usage
# similarity_metric = APIBasedSimilarity(
#     model="text-embedding-ada-002",
#     max_tokens=8191  # Optional: override default token limit
# )
# score, was_truncated = similarity_metric.calculate_similarity("text1", "text2")
# print(f"Similarity Score: {score} (Truncated: {was_truncated})")
