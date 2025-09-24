from __future__ import annotations
from typing import Protocol, List, Tuple

class DescriptorLearner(Protocol):
    def learn(self, text: str) -> List[Tuple[str, str, list[str]]]:
        """Return list of (name, description, aliases)"""
