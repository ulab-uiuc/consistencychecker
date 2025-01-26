from typing import Any, Dict, List, Tuple, Union

import litellm
import yaml
from tqdm import tqdm

from llmcheck.core.generator import BenchmarkGenerator
from llmcheck.core.tree import EvaluationTree
from llmcheck.metrics.factory import SimilarityConfig, SimilarityFactory
from llmcheck.nodes.node import Node
from llmcheck.nodes.verifiable_function import VerifiableFunction


class LLMCheck:
    def __init__(self,
                 evaluator_model: str,
                 evaluatee_model: str,
                 similarity_config: Union[Dict[str, Any], SimilarityConfig],
                 evaluator_model_temperature: float,
                 evaluatee_model_temperature: float,
                 evaluator_api_base: str,
                 evaluatee_api_base: str,
                 max_depth: int,
                 n_operations: int,
                 operation_code_format_enforce_prompt: str,
                 llm_max_new_tokens: int,
                 retry_max: int,
                 time_limit: int) -> None:
        if not llm_max_new_tokens:
            raise ValueError("llm_max_new_tokens must be set.")
        if not retry_max:
            raise ValueError("retry_max must be set.")
        self.evaluator_model = evaluator_model
        self.evaluatee_model = evaluatee_model
        self.max_depth = max_depth
        self.n_operations = n_operations
        print(f"[INFO] Evaluator API base: {evaluator_api_base}")
        print(f"[INFO] Target API base: {evaluatee_api_base}")
        self.evaluator_api_base = evaluator_api_base
        self.evaluatee_api_base = evaluatee_api_base
        self.evaluator_model_temperature = evaluator_model_temperature
        self.evaluatee_model_temperature = evaluatee_model_temperature
        self.bench_generator = BenchmarkGenerator(evaluator_model, evaluator_api_base, evaluator_model_temperature, llm_max_new_tokens)
        self.similarity_metric = SimilarityFactory.create_metric(similarity_config)
        self.operation_code_format_enforce_prompt = operation_code_format_enforce_prompt
        self.llm_max_new_tokens = llm_max_new_tokens
        self.retry_max = retry_max
        self.time_limit = time_limit

    def generate_root_content(self, constraints: str) -> Dict[str, Any]:
        response = litellm.completion(
            model=self.evaluator_model,
            messages=[{"role": "user", "content": constraints}],
            api_base=self.evaluator_api_base,
            temperature=self.evaluator_model_temperature,
            max_tokens=self.llm_max_new_tokens
        )
        response_str = response.choices[0].message.content
        response_dict = self._yaml_str_to_dict(response_str)
        return response_dict

    def evaluate(self, constraints: str, prompt_template: str, distance: List[int], root: Dict[str, Any], operations: List[Tuple[str, str]]) -> Dict[str, Any]:
        # test rood node
        retry: int = 0
        retry_max: int = self.retry_max
        state: str = ''
        print("[INFO] It is normal for errors and retries to occur when using LLM-generated YAML content and programs.")
        while retry <= retry_max:
            # if build vf and exec failed, make a new root
            try:
                if root:
                    root_content = root
                    print(f"[INFO] Overriding root content with set value: {root_content}")
                else:
                    root_content = self.generate_root_content(constraints)
                state = 'Root node generated'
                root_vf: VerifiableFunction = VerifiableFunction(**root_content, time_limit=self.time_limit)
                state = 'Root node verified'
                root_vf.exec(catch=False)
                state = 'Root node executable'
                tree = EvaluationTree(root_content)
                state = 'Root node yaml valid'

                if len(operations) >= self.n_operations:
                    operations = operations[:self.n_operations]
                    print(f"[INFO] Overriding operations with set value: {operations}")
                else:
                    root_code = root_content["code"]
                    prompt = prompt_template.format(n_operations=self.n_operations, root_code=root_code) ######
                    operations = self.bench_generator.generate_operations(prompt, self.n_operations)
                state = 'Operations generated'
                self._build_tree(tree.root, operations, 0)
                state = 'Tree built'
                break

            except Exception as e:
                print(f"[DEBUG] Goes as far as: {state}")
                print(f"[ERROR] {e}")
                print(f"[INFO] Retry {retry + 1}/{retry_max}")
                retry += 1


        metrics = self._calculate_metrics(tree, distance)

        tree_dict = self._tree_to_dict(tree.root)

        return {
            "evaluator_model": {
                "model": self.evaluator_model,
                "temperature": self.evaluator_model_temperature,
                "api_base": self.evaluator_api_base
            },
            "evaluatee_model": {
                "model": self.evaluatee_model,
                "temperature": self.evaluatee_model_temperature,
                "api_base": self.evaluatee_api_base
            },
            "root_content": root_content,
            "operations": operations,
            "metrics": metrics,
            "tree": tree_dict
        }

    def _build_tree(self, node: Node, operations: List[Tuple[str, str]], depth: int) -> Node:
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
                    current_node.content["exec_results"] = root_vf.exec(catch=True)

                for transform, reverse in operations:
                    current_node_dict = current_node.content
                    current_node_code = current_node_dict["code"]
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
                    middle_state_dict_updated["exec_results"] = middle_state_vf.exec(catch=True)
                    middle_state = middle_state_dict_updated
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
                    final_state_dict_updated["exec_results"] = final_state_vf.exec(catch=True)
                    final_state = final_state_dict_updated
                    child = current_node.add_child(
                        content=final_state,
                        middle_state=middle_state,
                        operation=(transform, reverse)
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
                        "Please apply the following operation to the text:\n"
                        f"Operation: {operation}\n{tail_prompt}\n"
                        f"Text: {content}\n"
                        f"Please do not include anything other than the transformed text."
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
                        "Please apply the following operation to the text:\n"
                        f"Operation: {operation}\n"
                        f"Text: {content}\n"
                        f"Please do not include anything other than the transformed text."
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
            "content": node.content,
            "operation": node.operation,
            "middle_state": node.middle_state,
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
                similarities = []
                with tqdm(total=len(node_pairs), desc=f"L-{dist} AVG Similarity") as pbar:
                    for a, b in node_pairs:
                        # remove JSON code block and elicit JSON string
                        a_exec_results_str: str = f"{a.content['exec_results']}"
                        b_exec_results_str: str = f"{b.content['exec_results']}"
                        similarities.append(self.similarity_metric.calculate_similarity(a_exec_results_str, b_exec_results_str))
                        pbar.update(1)
                metric_result[f"L-{dist} AVG"] = sum(similarities)/len(similarities)
                metric_result[f"L-{dist}"] = similarities
            else:
                metric_result[f"L-{dist} AVG"] = 0.0
                metric_result[f"L-{dist}"] = []

        return metric_result
