from typing import Dict, Optional, Union
from pydantic import BaseModel
from llmcheck.metrics.base import BaseSimilarityMetric
from llmcheck.metrics.huggingface import HuggingFaceSimilarity
from llmcheck.metrics.api import APIBasedSimilarity

class SimilarityConfig(BaseModel):
    type: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    device: Optional[str] = None
    similarity_prompt_template: Optional[str] = None

class SimilarityFactory:
    @staticmethod
    def create_metric(config: Union[Dict, SimilarityConfig]) -> BaseSimilarityMetric:
        if isinstance(config, dict):
            config = SimilarityConfig(**config)
            
        if config.type == "huggingface":
            return HuggingFaceSimilarity(
                model_name=config.model_name,
                device=config.device
            )
        elif config.type == "api":
            return APIBasedSimilarity(
                model=config.model_name,
                api_base=config.api_base
            )
        else:
            raise ValueError(f"Unknown similarity metric type: {config.type}")