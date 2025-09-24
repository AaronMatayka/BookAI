from __future__ import annotations
from src.ports.prompt_builder import PromptBuilder
from src.adapters.context_bank_store import ContextBankStore

class PromptEnricher(PromptBuilder):
    """Wraps ContextBank.enrich to keep the pipeline agnostic."""
    def __init__(self, context_store: ContextBankStore):
        self._store = context_store

    def build(self, base_prompt: str, page_text: str) -> str:
        return self._store.enrich(base_prompt, page_text)
