from typing import Any, Dict, List, Tuple, Union

import litellm
from tqdm import tqdm

from llmcheck.core.operations import OperationGenerator
from llmcheck.core.tree import EvaluationTree, Node
from llmcheck.metrics.factory import SimilarityConfig, SimilarityFactory


class LLMCheck:
    def __init__(self,
                 evaluator_model: str,
                 target_model: str,
                 similarity_config: Union[Dict[str, Any], SimilarityConfig],
                 evaluator_api_base: str = "",
                 target_api_base: str = "",
                 max_depth: int = 3,
                 n_operations: int = 3):
        self.evaluator_model = evaluator_model
        self.target_model = target_model
        self.max_depth = max_depth
        self.n_operations = n_operations
        if evaluator_api_base:
            print(f"[INFO] Overriding evaluator API base with set value: {evaluator_api_base}")
        if target_api_base:
            print(f"[INFO] Overriding target API base with set value: {target_api_base}")
        self.evaluator_api_base = evaluator_api_base
        self.target_api_base = target_api_base
        self.op_generator = OperationGenerator(evaluator_model)
        self.similarity_metric = SimilarityFactory.create_metric(similarity_config)

    def generate_root_content(self, constraints: str) -> str:
        if self.evaluator_api_base:
            response = litellm.completion(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": constraints}],
                api_base=self.evaluator_api_base
            )
        else:
            response = litellm.completion(
                model=self.evaluator_model,
                messages=[{"role": "user", "content": constraints}]
            )
        response_str = response.choices[0].message.content
        assert isinstance(response_str, str)
        return response_str

    def evaluate(self, constraints: str, distance: List[int], root: str = "", operations: List[Tuple[str, str]] = []) -> Dict[str, Any]:
        if root:
            root_content = root
            print(f"[INFO] Overriding root content with set value: {root_content}")
        else:
            root_content = self.generate_root_content(constraints)
        tree = EvaluationTree(root_content)

        if len(operations) >= self.n_operations:
            operations = operations[:self.n_operations]
            print(f"[INFO] Overriding operations with set value: {operations}")
        else:
            operations = self.op_generator.generate_operations(self.n_operations)

        self._build_tree(tree.root, operations, 0)

        # tree_str = self._str_tree(tree.root)

        metrics = self._calculate_metrics(tree, distance)

        return {
            "tree": tree,
            "root_content": root_content,
            "operations": operations,\
            "metrics": metrics
        }

    def _build_tree(self, node: Node, operations: List[Tuple[str, str]], depth: int) -> None:
        if depth >= self.max_depth:
            return

        nodes_to_process = [(node, depth)]

        total_nodes = sum(len(operations) ** i for i in range(self.max_depth - depth + 1)) - 1
        with tqdm(total=total_nodes, desc="Building tree") as pbar:
            while nodes_to_process:
                current_node, current_depth = nodes_to_process.pop(0)
                if current_depth >= self.max_depth:
                    continue

                for transform, reverse in operations:
                    # Apply transform to get middle state
                    middle_state = self._apply_operation(current_node.content, transform)
                    # Apply reverse to get final state
                    final_state = self._apply_operation(middle_state, reverse)

                    child = current_node.add_child(
                        content=final_state,
                        middle_state=middle_state,
                        operation=(transform, reverse)
                    )

                    nodes_to_process.append((child, current_depth + 1))
                    pbar.update(1)

    def _apply_operation(self, content: str, operation: str) -> str:
        if self.target_api_base:
            response = litellm.completion(
                model=self.target_model,
                messages=[
                    {"role": "user", "content": (
                        "Please apply the following operation to the text:\n"
                        f"Operation: {operation}\n"
                        f"Text: {content}\n"
                        f"Please do not include anything other than the transformed text."
                    )}
                ],
                api_base=self.target_api_base
            )
        else:
            response = litellm.completion(
                model=self.target_model,
                messages=[
                    {"role": "user", "content": (
                        "Please apply the following operation to the text:\n"
                        f"Operation: {operation}\n"
                        f"Text: {content}\n"
                        f"Please do not include anything other than the transformed text."
                    )}
                ]
            )
        response_str = response.choices[0].message.content
        assert isinstance(response_str, str)
        return response_str

    def _str_tree(self, node: Node, prefix: str = "", is_last: bool = True) -> str:
        result = ""
        marker = "└─ " if is_last else "├─ "
        result += prefix + marker + f"Content: {node.content}\n"
        if node.operation and node.middle_state:
            transform, reverse = node.operation
            child_prefix = prefix + ("   " if is_last else "│  ")
            result += child_prefix + f"Transform: {transform}\n"
            result += child_prefix + f"Middle state: {node.middle_state}\n"
            result += child_prefix + f"Reverse: {reverse}\n"
        new_prefix = prefix + ("   " if is_last else "│  ")
        for i, child in enumerate(node.children):
            result += self._str_tree(child, new_prefix, i == len(node.children) - 1)
        return result

    def _gather_all_nodes(self, root: Node) -> List[Node]:
        result = []
        queue = [root]
        while queue:
            current = queue.pop(0)
            result.append(current)
            queue.extend(current.children)
        return result

    def _calculate_metrics(self, tree: EvaluationTree, distance: List[int] = [1, 2, 3]) -> Dict[str, Any]:

        metric_result: Dict[str, Any] = {}
        for dist in distance:
            node_pairs = []
            for start_node in self._gather_all_nodes(tree.root):
                queue = [(start_node, 0)]
                while queue:
                    node, depth = queue.pop(0)
                    if depth == dist:
                        node_pairs.append((start_node, node))
                    elif depth < dist:
                        for child in node.children:
                            queue.append((child, depth + 1))

            if node_pairs:
                similarities = []
                with tqdm(total=len(node_pairs), desc=f"L-{dist} AVG Similarity") as pbar:
                    for a, b in node_pairs:
                        similarities.append(self.similarity_metric.calculate_similarity(a.content, b.content))
                        pbar.update(1)
                metric_result[f"L-{dist} AVG"] = sum(similarities)/len(similarities)
                metric_result[f"L-{dist}"] = similarities
            else:
                metric_result[f"L-{dist} AVG"] = 0.0
                metric_result[f"L-{dist}"] = []

        return metric_result
