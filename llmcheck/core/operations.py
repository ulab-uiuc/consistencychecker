from typing import List, Tuple

import litellm


class OperationGenerator:
    def __init__(self, evaluator_model: str):
        self.model = evaluator_model

    def generate_operations(self, n_operations: int) -> List[Tuple[str, str]]:
        prompt = f"""Generate {n_operations} pairs of transform-reverse operations for testing language model consistency.
        Each operation should modify the text and its reverse should restore it.
        Format each line as: "transform operation | reverse operation"
        Example: "translate to French | translate back to English"
        "translate to Japanese | translate back to English"
        Also, it can be: "reverse each sentence to get the opposite meaning | reverse each sentence to get the opposite meaning"
        Or: "summarize the text in extremely short form | expand the summary to a normal text"
        Please try out what I have suggested above and then come up with your own ideas. Please avoid overly simple operations like "add a period | remove a period" or "capitalize the first letter | lowercase the first letter".
        Please start each line without '- ', '1. ', 'a. ', 'i. ', etc. Keep it simple and clear. Just the operation and its reverse.
        """

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
