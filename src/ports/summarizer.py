from __future__ import annotations
from typing import Protocol

class Summarizer(Protocol):
    def summarize(self, text: str, token_cap: int) -> str: ...
