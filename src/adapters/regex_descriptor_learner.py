# src/adapters/regex_descriptor_learner.py
from __future__ import annotations
from typing import List, Tuple
from src.domain.context import DescriptorExtractor  # ← updated import
from src.ports.descriptor_learner import DescriptorLearner

class RegexDescriptorLearner(DescriptorLearner):
    def __init__(self):
        self._ex = DescriptorExtractor()

    def learn(self, text: str) -> List[Tuple[str, str, list[str]]]:
        results = []
        for d in self._ex.suggest(text):
            results.append((d.name, d.description, d.aliases))
        return results
