# src/adapters/ollama_summarizer.py
from __future__ import annotations
from src.ports.summarizer import Summarizer
from src.adapters.ollama_api import ollama_generate_prompt  # ← updated import

class OllamaSummarizer(Summarizer):
    def __init__(self, model: str, url: str, context_supplier):
        self.model = model
        self.url = url
        self.context_supplier = context_supplier  # callable: page_text -> context_snippet

    def summarize(self, text: str, token_cap: int) -> str:
        context = self.context_supplier(text)
        return ollama_generate_prompt(self.model, context, text, self.url) or ""
