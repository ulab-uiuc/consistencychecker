from typing import List, Tuple

import litellm


class OperationGenerator:
    def __init__(self, evaluator_model: str):
        self.model = evaluator_model

    def generate_operations(self, prompt: str, n_operations: int) -> List[Tuple[str, str]]:
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
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
