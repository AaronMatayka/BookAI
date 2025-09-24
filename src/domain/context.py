# src/domain/context.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import json
import re

# Try to import spaCy, but make it optional.
try:
    import spacy
    from spacy.tokens import Doc, Span, Token
except ImportError:
    spacy = None
    Doc = Span = Token = None


@dataclass
class Descriptor:
    """A simple data structure to hold an extracted piece of information."""
    name: str
    description: str
    aliases: List[str]


class ContextBank:
    """Manages loading, saving, and querying a persistent JSON file of learned context."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Dict] = {"entities": {}}
        self._compiled_regex: Optional[re.Pattern] = None
        self.load()

    def load(self) -> None:
        """Loads context from the JSON file, creating it if it doesn't exist."""
        try:
            if self.path.exists():
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.save()
        except (json.JSONDecodeError, IOError):
            self.data = {"entities": {}}
            self.save()
        self._compiled_regex = None

    def save(self) -> None:
        """Saves the current context to the JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self) -> None:
        """Clears all learned context."""
        self.data = {"entities": {}}
        self._compiled_regex = None
        self.save()

    def upsert(self, name: str, description: str, aliases: Optional[List[str]] = None) -> None:
        """
        Adds or updates an entity in the context bank (merge by similarity, keep aliases & dedupe descriptions).
        """
        name = (name or "").strip()
        if not name:
            return

        aliases = [a.strip() for a in aliases or [] if a and a.strip() and a.strip().lower() != name.lower()]

        def find_similar(target: str, alias_list: List[str]) -> Optional[str]:
            t_lower = target.lower()
            candidates = {t_lower}
            candidates.update(a.lower() for a in alias_list)
            for existing, meta in self.data.get("entities", {}).items():
                existing_lower = existing.lower()
                if existing_lower in candidates:
                    return existing
                if existing_lower.split()[0] == t_lower.split()[0]:
                    return existing
                if existing_lower in t_lower or t_lower in existing_lower:
                    return existing
                for al in meta.get("aliases", []):
                    al_lower = al.lower()
                    if al_lower in candidates or al_lower == t_lower:
                        return existing
                    if al_lower in t_lower or t_lower in al_lower:
                        return existing
            generic_labels = {"girl", "boy", "woman", "man"}
            t_words = set(re.split(r"\W+", t_lower))
            target_generics = generic_labels & t_words
            if target_generics:
                for existing_name, meta in self.data.get("entities", {}).items():
                    e_words = set(re.split(r"\W+", existing_name.lower()))
                    existing_generics = generic_labels & e_words
                    if existing_generics and existing_generics == target_generics:
                        return existing_name
            return None

        canonical = find_similar(name, aliases)
        name_key = canonical or name

        entity = self.data.setdefault("entities", {}).setdefault(name_key, {"aliases": [], "description": ""})

        current_aliases: set[str] = set(meta.strip() for meta in entity.get("aliases", []))
        if name_key != name:
            current_aliases.add(name)
        current_aliases.update(a for a in aliases if a and a.lower() != name_key.lower())
        current_aliases.discard(name_key)

        new_desc = (description or "").strip()
        if new_desc:
            existing_descs = set(d.strip() for d in str(entity.get("description", "")).split(';') if d.strip())
            existing_descs.add(new_desc)
            generic_keywords = [
                "no specific description", "no specific physical description",
                "no specific visual descriptors", "no physical description",
                "not described"
            ]
            has_specific = any(
                not any(gk in d.lower() for gk in generic_keywords) for d in existing_descs
            )
            if has_specific:
                existing_descs = {
                    d for d in existing_descs
                    if not any(gk in d.lower() for gk in generic_keywords)
                }
            entity["description"] = "; ".join(sorted(list(existing_descs)))

        if current_aliases:
            entity["aliases"] = sorted(list(current_aliases))

        self._compiled_regex = None

    def _get_regex(self) -> re.Pattern:
        """Compiles and caches a regex to find all known entity names and aliases in text."""
        if self._compiled_regex is not None:
            return self._compiled_regex

        names: List[str] = []
        for base, meta in self.data.get("entities", {}).items():
            names.append(base)
            names.extend(meta.get("aliases", []))

        parts = sorted({re.escape(n) for n in names if n.strip()}, key=len, reverse=True)
        if not parts:
            self._compiled_regex = re.compile(r"$a")
        else:
            self._compiled_regex = re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)
        return self._compiled_regex

    def matches_in_text(self, text: str) -> List[str]:
        """Finds all occurrences of known entities in a given text."""
        rx = self._get_regex()
        return sorted(list(set(m.group(0).strip() for m in rx.finditer(text or ""))), key=text.lower().find)

    def build_context_snippet(self, matches: List[str]) -> str:
        """Creates a formatted string snippet of context for given matches."""
        if not matches:
            return ""
        canonical_map: Dict[str, str] = {}
        for base, meta in self.data.get("entities", {}).items():
            for alias in [base] + meta.get("aliases", []):
                canonical_map[alias.lower()] = base
        bases_in_order = []
        seen_bases = set()
        for match in matches:
            base_name = canonical_map.get(match.lower())
            if base_name and base_name not in seen_bases:
                bases_in_order.append(base_name)
                seen_bases.add(base_name)
        parts = []
        for base in bases_in_order:
            desc = str(self.data["entities"].get(base, {}).get("description", "")).strip()
            if desc:
                parts.append(f"{base} ({desc})")
        return "; ".join(parts)

    def relevant_snippet(self, page_text: str) -> str:
        """Finds entities on a page and returns their relevant context snippet."""
        return self.build_context_snippet(self.matches_in_text(page_text or ""))


class DescriptorExtractor:
    """
    Extracts character/location descriptors from text using spaCy (if available)
    or a regex fallback.
    """
    _VISUAL_KEYWORDS = {
        "hair", "eye", "eyes", "beard", "freckle", "scar", "tattoo", "tall", "short",
        "slim", "thin", "stocky", "broad", "gaunt", "muscular", "built", "slender",
        "pale", "fair", "dark", "tan", "olive", "red", "blonde", "brown", "black",
        "blue", "green", "hazel", "grey", "gray", "golden", "face", "jaw", "cheek",
        "nose", "lips", "mouth", "soft", "sharp", "angular", "jacket", "coat",
        "dress", "boots", "jeans", "leather", "hoodie", "shirt", "gown", "cape",
        "young", "teenage", "fifteen", "old", "ancient", "slender", "lithe", "glasses",
        "glasses",
    }

    def __init__(self) -> None:
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except IOError:
                print("spaCy model 'en_core_web_sm' not found. Falling back to regex.")
                print("For better results, run: python -m spacy download en_core_web_sm")

    def suggest(self, text: str) -> List[Descriptor]:
        if not text:
            return []
        items = self._suggest_spacy(text) if self.nlp else self._suggest_regex(text)
        return self._dedupe(items)

    def _is_visual(self, s: str) -> bool:
        return any(f' {k} ' in f' {s.lower()} ' for k in self._VISUAL_KEYWORDS)

    def _suggest_spacy(self, text: str) -> List[Descriptor]:
        out: List[Descriptor] = []
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ not in ["PERSON", "GPE", "LOC", "ORG"]:
                continue

            description = self._extract_spacy_description_for_entity(ent)
            if description and self._is_visual(description):
                out.append(Descriptor(ent.text, description, self._default_aliases(ent.text)))

            root_token = ent.root
            for child in root_token.children:
                if child.dep_ == 'poss':
                    possessed_noun = child
                    possessed_desc = self._extract_spacy_description_for_entity(possessed_noun)
                    if possessed_desc and self._is_visual(possessed_desc):
                        full_desc = f"{possessed_noun.text} {possessed_desc}"
                        out.append(Descriptor(ent.text, full_desc, self._default_aliases(ent.text)))
        return out

    def _extract_spacy_description_for_entity(self, ent_or_token: Span | Token) -> str:
        if hasattr(ent_or_token, 'rights'):
            for token in ent_or_token.rights:
                if token.dep_ == "appos":
                    appos_phrase = " ".join([t.text for t in token.subtree])
                    return appos_phrase.strip()

        token_root = ent_or_token.root if isinstance(ent_or_token, Span) else ent_or_token
        verb_head = token_root.head
        if verb_head.lemma_ in ("be", "seem", "look", "appear"):
            full_desc_parts = []
            for child in verb_head.children:
                if child.dep_ in ("attr", "acomp"):
                    desc_phrase = " ".join(t.text for t in child.subtree)
                    full_desc_parts.append(desc_phrase)
            if full_desc_parts:
                return " ".join(full_desc_parts)
        return ""

    def _suggest_regex(self, text: str) -> List[Descriptor]:
        out: List[Descriptor] = []
        last_person: Optional[str] = None

        name_pattern = r"\b([A-Z][a-z']+(?:\s[A-Z][a-z']+)?)\b"
        patterns = [
            re.compile(rf"{name_pattern}\s*,\s*(?:a|an|the)?\s*([a-zA-Z\s\-]+?),", re.IGNORECASE),
            re.compile(rf"{name_pattern}\s+(?:is|was|had|wore)\s+([^\.]+)\.", re.IGNORECASE),
            re.compile(rf"(He|She)\s+(?:is|was|had|wore)\s+([^\.]+)\.", re.IGNORECASE),
            re.compile(rf"{name_pattern}[^\.]*?\b(?:in|with)\s+([^\,\.]+)", re.IGNORECASE),
            re.compile(r"(His|Her)[^\.]*?\bhair\s+(?:was|is|were|looked|appeared|seemed)?\s*([^\,\.]+)", re.IGNORECASE),
            re.compile(r"(His|Her)\s+glasses\s+(?:were|are|looked|appeared|seemed|perched)?\s*([^\,\.]+)", re.IGNORECASE),
        ]

        for line in text.split('\n'):
            for pat in patterns:
                for match in pat.finditer(line):
                    name, desc = match.groups()
                    lower_name = (name or "").lower()
                    is_pronoun = lower_name in ('he', 'she', 'his', 'her')
                    if not is_pronoun and not name[0].isupper():
                        continue
                    if is_pronoun:
                        if last_person and self._is_visual(desc):
                            out.append(Descriptor(last_person, desc.strip(), []))
                    else:
                        if self._is_visual(desc):
                            n = name.strip()
                            out.append(Descriptor(n, desc.strip(), self._default_aliases(n)))
                        last_person = name.strip()
        return out

    @staticmethod
    def _default_aliases(name: str) -> List[str]:
        parts = name.split()
        return [parts[0]] if len(parts) > 1 else []

    @staticmethod
    def _dedupe(items: List[Descriptor]) -> List[Descriptor]:
        merged: Dict[str, Descriptor] = {}
        for d in items:
            name_lower = d.name.lower()
            if name_lower not in merged:
                merged[name_lower] = d
            else:
                existing = merged[name_lower]
                existing_descs = set(s.strip() for s in existing.description.split(';') if s.strip())
                existing_descs.add(d.description.strip())
                existing.description = "; ".join(sorted(list(existing_descs)))

                existing_aliases = set(existing.aliases)
                existing_aliases.update(d.aliases)
                existing.aliases = sorted(list(existing_aliases))
        return list(merged.values())


def enrich_prompt_with_context(bank: ContextBank, base_prompt: str, page_text: str) -> str:
    """Adds relevant context to a base prompt."""
    snippet = bank.relevant_snippet(page_text or "")
    return f"In a scene where {snippet}: {base_prompt}" if snippet else base_prompt
