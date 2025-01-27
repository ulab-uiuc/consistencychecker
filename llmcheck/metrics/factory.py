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
    @staticmethod
    def create_metric(config: Union[Dict[str, Any], SimilarityConfig]) -> List[BaseSimilarityMetric]:
        if isinstance(config, dict):
            config = SimilarityConfig(**config)

        if config.type == "huggingface":
            return [
                HuggingFaceSimilarity(
                    model_name=config.model_name,
                    device=config.device
                ),
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
