from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from llmcheck.metrics.base import BaseSimilarityMetric


class HuggingFaceSimilarity(BaseSimilarityMetric):
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=True,
                              return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokens)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            assert isinstance(embeddings, torch.Tensor)
        return embeddings

    def calculate_similarity(self, text1: str, text2: str) -> float:
        embeddings = self._get_embeddings([text1, text2])
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0], embeddings[1], dim=0
        )
        return_float = similarity.item()
        assert isinstance(return_float, float)
        return return_float

    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        if len(texts1) != len(texts2):
            raise ValueError("Input text lists must have same length")

        embeddings1 = self._get_embeddings(texts1)
        embeddings2 = self._get_embeddings(texts2)

        similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        similarities_list = similarities.tolist()
        assert isinstance(similarities_list, list)
        return similarities_list
