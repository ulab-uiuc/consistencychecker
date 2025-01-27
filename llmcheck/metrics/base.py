from abc import ABC, abstractmethod
from typing import List


class BaseSimilarityMetric(ABC):
    name: str = ""
    @abstractmethod
    def calculate_similarity(self, text1: str, text2: str) -> float:
        pass

    @abstractmethod
    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[float]:
        pass
