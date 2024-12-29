from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Node:
    content: str
    operation: Optional[tuple[str, str]] = None  # (transform, reverse)
    middle_state: Optional[str] = None  # Store the intermediate transformed state
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    def add_child(self, content: str, middle_state: str, operation: tuple[str, str]) -> 'Node':
        child = Node(
            content=content,
            operation=operation,
            middle_state=middle_state,
            parent=self
        )
        self.children.append(child)
        return child

class EvaluationTree:
    def __init__(self, root_content: str):
        self.root = Node(content=root_content)

    def add_child(self, parent: Node, content: str, middle_state: str, operation: tuple[str, str]) -> Node:
        child = Node(
            content=content,
            operation=operation,
            middle_state=middle_state,
            parent=parent
        )
        parent.children.append(child)
        return child
