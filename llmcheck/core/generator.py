from typing import Any, Dict, List, Tuple

import litellm
import yaml

from llmcheck.core.tree import EvaluationTree
from llmcheck.nodes.verifiable_function import VerifiableFunction


class BenchmarkGenerator:
    def __init__(self, evaluator_model: str, evaluator_model_api_base: str, evaluator_model_temperature: float, llm_max_new_tokens: int, time_limit: float):
        self.model = evaluator_model
        self.api_base = evaluator_model_api_base
        self.temperature = evaluator_model_temperature
        self.llm_max_new_tokens = llm_max_new_tokens
        self.time_limit = time_limit

    def _yaml_str_to_dict(self, yaml_str: str) -> Dict[str, Any]:
        yaml_str_trimmed = yaml_str.strip("```yaml").strip("```yml").strip("```").strip("\n")
        result_dict: Dict[str, Any] = yaml.safe_load(yaml_str_trimmed)
        return result_dict

    def _generate_root_content(self, constraints: str) -> Dict[str, Any]:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": constraints}],
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.llm_max_new_tokens
        )
        response_str = response.choices[0].message.content
        response_dict = self._yaml_str_to_dict(response_str)
        # verify validity of the response (static)
        EvaluationTree(response_dict)
        # verify validity of the response (execution)
        root_vf: VerifiableFunction = VerifiableFunction(**response_dict, time_limit=self.time_limit)
        root_vf.exec(catch=False)
        # stringify each element in inputs
        return response_dict

    def _generate_operations(self, prompt: str, n_operations: int) -> List[List[str]]:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.llm_max_new_tokens
        )

        operations = []
        lines = response.choices[0].message.content.strip().split('\n')
        for line in lines:
            if '|' not in line:
                continue
            transform, reverse = line.split('|')
            operations.append([transform.strip(), reverse.strip()])
        if len(operations) < n_operations:
            raise ValueError(f"Could not generate {n_operations} operations. Only {len(operations)} operations were generated.")

        return operations[:n_operations]

    def generate_benchmark(
            self,
            constraints: str,
            prompt: str,
            n_operations: int
    ) -> Tuple[Dict[str, Any], List[List[str]]]:
        root = self._generate_root_content(constraints)
        operations = self._generate_operations(prompt, n_operations)
        return root, operations
