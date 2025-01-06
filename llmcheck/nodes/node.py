import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Node:
    content: Dict[str, Any]
    operation: Optional[tuple[str, str]] = None  # (transform, reverse)
    middle_state: Optional[Dict[str, Any]] = None
    parent: Optional['Node'] = None
    children: List['Node'] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    def add_child(self, content: Dict[str, Any], middle_state: Dict[str, Any], operation: tuple[str, str]) -> 'Node':
        child = Node(
            content=content,
            operation=operation,
            middle_state=middle_state,
            parent=self
        )
        self.children.append(child)
        return child