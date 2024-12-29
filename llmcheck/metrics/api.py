from typing import List

import litellm
import numpy as np

from llmcheck.metrics.base import BaseSimilarityMetric


class APIBasedSimilarity(BaseSimilarityMetric):
    def __init__(self,
                 model: str = "text-embedding-ada-002",
                 api_base: str = ""):
        self.model = model
        if api_base:
            print(f"[INFO] Overriding API embedding model with set value: {model}")
        self.api_base = api_base

    def _get_embedding(self, text: str) -> np.ndarray[float, np.dtype[np.float64]]:
        """Fetch the embedding vector for a given text using the specified model."""
        if self.api_base:
            response = litellm.embedding(
                model=self.model,
                input=text,
                api_base=self.api_base
            )
        else:
            response = litellm.embedding(
                model=self.model,
                input=text,
            )

        try:
            embedding = response['data'][0]['embedding']
            return np.array(embedding)
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to retrieve embedding: {e}")

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate the cosine similarity between embeddings of two texts."""
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
        """Calculate similarities for multiple pairs of texts."""
        if len(texts1) != len(texts2):
            raise ValueError("Input lists must have the same length.")

        similarities = []
        for t1, t2 in zip(texts1, texts2):
            similarities.append(self.calculate_similarity(t1, t2))
        return similarities

# Example Usage
# similarity_metric = EmbeddingBasedSimilarity(api_key="your_api_key")
# score = similarity_metric.calculate_similarity("text1", "text2")
# print(f"Similarity Score: {score}")
