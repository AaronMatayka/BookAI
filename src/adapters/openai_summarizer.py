from __future__ import annotations
from src.ports.summarizer import Summarizer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class OpenAISummarizer(Summarizer):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def summarize(self, text: str, token_cap: int) -> str:
        if not self.api_key or OpenAI is None:
            return ""
        client = OpenAI(api_key=self.api_key)
        system = (
            "You are an expert at creating visual prompts from literary text. "
            f"Summarize the page into one concise paragraph with concrete visual details (~{token_cap} words)."
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":system},{"role":"user","content":f"TEXT:\n{text}"}],
                temperature=0.2,
                max_tokens=200,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""
