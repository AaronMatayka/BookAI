# src/adapters/context_bank_store.py
from __future__ import annotations
from pathlib import Path

from src.domain.context import ContextBank, enrich_prompt_with_context  # ← updated import
from src.ports.context_store import ContextStore

class ContextBankStore(ContextStore):
    def __init__(self, bank_path: Path):
        self.bank = ContextBank(bank_path)

    def reset(self) -> None:
        self.bank.reset()

    def upsert(self, name: str, description: str, aliases: list[str]) -> None:
        self.bank.upsert(name, description, aliases)

    def save(self) -> None:
        self.bank.save()

    def relevant_snippet(self, page_text: str) -> str:
        return self.bank.relevant_snippet(page_text)

    # Convenience used by PromptEnricher adapter
    def enrich(self, base_prompt: str, page_text: str) -> str:
        return enrich_prompt_with_context(self.bank, base_prompt, page_text=page_text)

    def export_snapshot(self, dest_path: str) -> None:
        import time
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# BookAI Context Snapshot",
            f"# Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#" + "-" * 50
        ]
        entities = self.bank.data.get("entities", {})
        if not entities:
            lines.append("\n# Context bank is empty.")
        else:
            for name, meta in sorted(entities.items()):
                desc = (meta.get("description") or "").strip()
                aliases = [a for a in (meta.get("aliases") or []) if a]
                line = f"## {name}\n"
                if desc:   line += f"- Description: {desc}\n"
                if aliases: line += f"- Aliases: {', '.join(aliases)}\n"
                lines.append(line)
        dest.write_text("\n".join(lines), encoding="utf-8")
