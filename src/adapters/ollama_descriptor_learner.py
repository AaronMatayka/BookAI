# src/adapters/ollama_descriptor_learner.py
from __future__ import annotations
from typing import List, Tuple
from src.adapters.ollama_api import ollama_extract_descriptors  # ← updated import
from src.ports.descriptor_learner import DescriptorLearner

class OllamaDescriptorLearner(DescriptorLearner):
    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url

    def learn(self, text: str) -> List[Tuple[str, str, list[str]]]:
        pairs = ollama_extract_descriptors(self.model, text, self.url) or []
        return [(name, desc, []) for (name, desc) in pairs]
