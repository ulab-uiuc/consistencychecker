from typing import List, Tuple

import litellm


class OperationGenerator:
    def __init__(self, evaluator_model: str, evaluator_model_api_base: str, evaluator_model_temperature: float):
        self.model = evaluator_model
        self.api_base = evaluator_model_api_base
        self.temperature = evaluator_model_temperature

    def generate_operations(self, prompt: str, n_operations: int) -> List[Tuple[str, str]]:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            api_base=self.api_base,
            temperature=self.temperature,
        )

        operations = []
        lines = response.choices[0].message.content.strip().split('\n')
        for line in lines:
            if '|' not in line:
                continue
            transform, reverse = line.split('|')
            operations.append((transform.strip(), reverse.strip()))
        if len(operations) < n_operations:
            raise ValueError(f"Could not generate {n_operations} operations. Only {len(operations)} operations were generated.")
        return operations
