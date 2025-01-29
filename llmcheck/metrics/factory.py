from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from llmcheck.metrics.api import APIBasedSimilarity
from llmcheck.metrics.base import BaseSimilarityMetric
from llmcheck.metrics.bleu import BLEUMetric
from llmcheck.metrics.huggingface import HuggingFaceSimilarity
from llmcheck.metrics.rouge import ROUGEMetric


class SimilarityConfig(BaseModel):
    type: str
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    device: Optional[str] = None
    similarity_prompt_template: Optional[str] = None

class SimilarityFactory:
    # Dictionary to cache models
    _huggingface_cache: Dict[str, HuggingFaceSimilarity] = {}

    @staticmethod
    def create_metric(config: Union[Dict[str, Any], SimilarityConfig]) -> List[BaseSimilarityMetric]:
        if isinstance(config, dict):
            config = SimilarityConfig(**config)

        # Check if the config type is huggingface
        if config.type == "huggingface":
            cache_key = f"{config.model_name}-{config.device}"

            # Check if the model is already cached
            if cache_key in SimilarityFactory._huggingface_cache:
                huggingface_model = SimilarityFactory._huggingface_cache[cache_key]
            else:
                # If not cached, load the model and cache it
                huggingface_model = HuggingFaceSimilarity(
                    model_name=config.model_name,
                    device=config.device
                )
                SimilarityFactory._huggingface_cache[cache_key] = huggingface_model

            # Return the list of metrics, including the cached HuggingFace model
            return [
                huggingface_model,
                BLEUMetric(),
                ROUGEMetric("rouge1"),
                ROUGEMetric("rouge2"),
                ROUGEMetric("rougeL"),
            ]

        elif config.type == "api":
            return [
                APIBasedSimilarity(
                    model=config.model_name,
                    api_base=config.api_base if config.api_base else ""
                ),
                BLEUMetric(),
                ROUGEMetric("rouge1"),
                ROUGEMetric("rouge2"),
                ROUGEMetric("rougeL"),
            ]
        else:
            raise ValueError(f"Unknown similarity metric type: {config.type}")
