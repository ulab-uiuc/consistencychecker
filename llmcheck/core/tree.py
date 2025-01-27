from typing import Any, Dict, List

from llmcheck.nodes.node import Node


class EvaluationTree:
    def __init__(self, root_content: Dict[str, Any]):
        self.root = Node(content=root_content)

    def add_child(self, parent: Node, content: Dict[str, Any], middle_state: Dict[str, Any], operation: List[str]) -> Node:
        child = Node(
            content=content,
            operation=operation,
            middle_state=middle_state,
            parent=parent
        )
        parent.children.append(child)
        return child
