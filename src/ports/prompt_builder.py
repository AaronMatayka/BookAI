from __future__ import annotations
from typing import Protocol

class PromptBuilder(Protocol):
    def build(self, base_prompt: str, page_text: str) -> str: ...
