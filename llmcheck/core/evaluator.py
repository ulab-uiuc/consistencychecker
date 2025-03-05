# defaultdict
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Union

import litellm
import yaml
from tqdm import tqdm

from llmcheck.core.tree import EvaluationTree
from llmcheck.metrics.factory import SimilarityConfig, SimilarityFactory
from llmcheck.nodes.node import Node
from llmcheck.nodes.verifiable_function import VerifiableFunction


class LLMCheck:
    def __init__(self,
                 evaluatee_model: str,
                 similarity_config: Union[Dict[str, Any], SimilarityConfig],
                 evaluatee_model_temperature: float,
                 evaluatee_api_base: str,
                 max_depth: int,
                 operation_code_format_enforce_prompt: str,
                 llm_max_new_tokens: int,
                 retry_max: int,
                 time_limit: int) -> None:
        if not llm_max_new_tokens:
            raise ValueError("llm_max_new_tokens must be set.")
        if not retry_max:
            raise ValueError("retry_max must be set.")
        self.evaluatee_model = evaluatee_model
        self.max_depth = max_depth
        print(f"[INFO] Target API base: {evaluatee_api_base}")
        self.evaluatee_api_base = evaluatee_api_base
        self.evaluatee_model_temperature = evaluatee_model_temperature
        self.similarity_metrics = SimilarityFactory.create_metric(similarity_config)
        self.operation_code_format_enforce_prompt = operation_code_format_enforce_prompt
        self.llm_max_new_tokens = llm_max_new_tokens
        self.retry_max = retry_max
        self.time_limit = time_limit

    def evaluate(self, distance: List[int], root: Dict[str, Any], operations: List[List[str]]) -> Dict[str, Any]:
        # test rood node
        retry: int = 0
        retry_max: int = self.retry_max
        state: str = ''
        # print("[INFO] It is normal for errors and retries to occur when using LLM-generated YAML content and programs.")
        while retry <= retry_max:
            # if build vf and exec failed, make a new root
            try:
                root_content = root
                root_content_abstract = f"{root_content}"[:100]
                print(f"[INFO] Overriding root content with set value: {root_content_abstract}...")
                state = 'Root node generated'
                root_vf: VerifiableFunction = VerifiableFunction(**root_content, time_limit=self.time_limit)
                state = 'Root node verified'
                root_vf.exec(catch=False)
                state = 'Root node executable'
                tree = EvaluationTree(root_content)
                state = 'Root node yaml valid'

                state = 'Operations generated'
                self._build_tree(tree.root, operations, 0)
                state = 'Tree built'
                break

            except Exception as e:
                print(f"[DEBUG] Goes as far as: {state}")
                print(f"[ERROR] {e}")
                print(f"[INFO] Retry {retry + 1}/{retry_max}")
                retry += 1

        if retry > retry_max:
            raise Exception(f"Failed to build tree after {retry_max} retries")

        metrics = self._calculate_metrics(tree, distance)

        tree_dict = self._tree_to_dict(tree.root)

        return {
            "operations": operations,
            "metrics": metrics,
            "evaluatee_model": {
                "model": self.evaluatee_model,
                "temperature": self.evaluatee_model_temperature,
                "api_base": self.evaluatee_api_base
            },
            "root_content": root_content,
            "tree": tree_dict
        }

    def _build_tree(self, node: Node, operations: List[List[str]], depth: int) -> Node:
        print(f"[INFO] Building tree with depth {self.max_depth} and {len(operations)} operations")
        if depth >= self.max_depth:
            raise ValueError(f"Depth {depth} exceeds the maximum depth {self.max_depth}")

        nodes_to_process = [(node, depth)]

        total_nodes = sum(len(operations) ** i for i in range(self.max_depth - depth + 1)) - 1
        with tqdm(total=total_nodes, desc="Building tree") as pbar:
            while nodes_to_process:
                current_node, current_depth = nodes_to_process.pop(0)
                if current_depth >= self.max_depth:
                    continue
                if current_depth == 0: # root
                    # execute the code
                    root_vf: VerifiableFunction = VerifiableFunction(**current_node.content, time_limit=self.time_limit)
                    current_node.content["exec_results"] = "\n".join([f"{result}" for result in root_vf.exec(catch=True)])

                for transform, reverse in operations:

                    current_node_dict = current_node.content
                    current_node_code = current_node_dict["code"]
                    while True:
                        try:
                            # Apply transform to get middle state
                            middle_state_code = self._apply_operation(current_node_code, transform, self.operation_code_format_enforce_prompt)
                            middle_state_code_programming_language = middle_state_code.split("\n")[0].strip('```').strip().lower()
                            # remove code block
                            middle_state_code_content = middle_state_code.split("\n", 1)[1].rsplit("```", 1)[0].strip("\n")
                            middle_state_dict_updated = current_node_dict.copy()
                            # drop exec_results
                            middle_state_dict_updated.pop("exec_results", None)
                            middle_state_dict_updated["code"] = middle_state_code_content
                            middle_state_dict_updated["programming_language"] = middle_state_code_programming_language
                            middle_state_vf: VerifiableFunction = VerifiableFunction(**middle_state_dict_updated, time_limit=self.time_limit)
                            middle_state_dict_updated["exec_results"] = "\n".join([f"{result}" for result in middle_state_vf.exec(catch=True)])
                            middle_state = middle_state_dict_updated
                            break
                        except Exception as e:
                            print(f"[ERROR] Error during middle state transformation: {e}. Retrying...")

                    while True:
                        try:
                            # Apply reverse to get final state
                            final_state_code = self._apply_operation(middle_state_code, reverse, self.operation_code_format_enforce_prompt)
                            final_state_code_programming_language = final_state_code.split("\n")[0].strip('```').strip().lower()
                            # remove code block
                            final_state_code_content = final_state_code.split("\n", 1)[1].rsplit("```", 1)[0].strip("\n")
                            final_state_dict_updated = current_node_dict.copy()
                            # drop exec_results
                            final_state_dict_updated.pop("exec_results", None)
                            final_state_dict_updated["code"] = final_state_code_content
                            final_state_dict_updated["programming_language"] = final_state_code_programming_language
                            final_state_vf: VerifiableFunction = VerifiableFunction(**final_state_dict_updated, time_limit=self.time_limit)
                            final_state_dict_updated["exec_results"] = "\n".join([f"{result}" for result in final_state_vf.exec(catch=True)])
                            final_state = final_state_dict_updated
                            break
                        except Exception as e:
                            print(f"[ERROR] Error during final state transformation: {e}. Retrying...")

                    child = current_node.add_child(
                        content=final_state,
                        middle_state=middle_state,
                        operation=[transform, reverse]
                    )

                    nodes_to_process.append((child, current_depth + 1))
                    pbar.update(1)
            return node

    def _apply_operation(self, content: str, operation: str, tail_prompt: str) -> str:
        if self.evaluatee_api_base:
            response = litellm.completion(
                model=self.evaluatee_model,
                messages=[
                    {"role": "user", "content": (
                        "Please apply the following operation to the program code:\n"
                        f"Operation: {operation}\n"
                        f"Program code: {content}\n"
                        f"Please do not include anything other than the transformed text.\n"
                        f"{tail_prompt}\n"
                    )}
                ],
                api_base=self.evaluatee_api_base,
                temperature=self.evaluatee_model_temperature,
                max_tokens=self.llm_max_new_tokens
            )
        else:
            response = litellm.completion(
                model=self.evaluatee_model,
                messages=[
                    {"role": "user", "content": (
                        "Please apply the following operation to the program code:\n"
                        f"Operation: {operation}\n"
                        f"Program code: {content}\n"
                        f"Please do not include anything other than the transformed text.\n"
                        f"{tail_prompt}\n"
                    )}
                ],
                temperature=self.evaluatee_model_temperature,
                max_tokens=self.llm_max_new_tokens
            )
        response_str = response.choices[0].message.content
        assert isinstance(response_str, str)
        return response_str


    def _tree_to_dict(self, node: Node) -> Dict[str, Any]:
        node_dict = {
            "operation": deepcopy(node.operation),
            "middle_state": deepcopy(node.middle_state),
            "content": deepcopy(node.content),
            "children": [self._tree_to_dict(child) for child in node.children]
        }
        return node_dict

    def _gather_all_nodes(self, root: Node) -> List[Node]:
        result = []
        queue = [root]
        while queue:
            current = queue.pop(0)
            result.append(current)
            queue.extend(current.children)
        return result

    def _yaml_str_to_dict(self, yaml_str: str) -> Dict[str, Any]:
        yaml_str_trimmed = yaml_str.strip("```yaml").strip("```yml").strip("```").strip("\n")
        result_dict: Dict[str, Any] = yaml.safe_load(yaml_str_trimmed)
        return result_dict

    def _calculate_metrics(self, tree: EvaluationTree, distance: List[int]) -> Dict[str, Any]:

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
                similarities = defaultdict(list)
                with tqdm(total=len(node_pairs), desc=f"L-{dist} AVG Similarity") as pbar:
                    for a, b in node_pairs:
                        # remove JSON code block and elicit JSON string
                        a_exec_results_str: str = a.content['exec_results']
                        b_exec_results_str: str = b.content['exec_results']
                        for metric in self.similarity_metrics:
                            similarities[metric.name].append(metric.calculate_similarity(a_exec_results_str, b_exec_results_str))
                        pbar.update(1)
                for metric_name in similarities:
                    metric_result[f"{metric_name} L-{dist} AVG"] = sum(similarities[metric_name]) / len(similarities[metric_name])
                    metric_result[f"{metric_name} L-{dist}"] = similarities[metric_name]
            else:
                for metric in self.similarity_metrics:
                    metric_name_iter: str = metric.name
                    metric_result[f"{metric_name_iter} L-{dist} AVG"] = 0.0
                    metric_result[f"{metric_name_iter} L-{dist}"] = []

        return metric_result
