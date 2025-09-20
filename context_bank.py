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
        Adds or updates an entity in the context bank.
        It intelligently merges descriptions to build a richer profile over time.
        """
        name = (name or "").strip()
        if not name:
            return

        entity = self.data.setdefault("entities", {}).setdefault(name, {"aliases": [], "description": ""})

        # Merge descriptions, avoiding duplicates
        new_desc = (description or "").strip()
        if new_desc:
            current_descs = set(d.strip() for d in str(entity.get("description", "")).split(';') if d.strip())
            current_descs.add(new_desc)
            entity["description"] = "; ".join(sorted(list(current_descs)))

        # Add new aliases, avoiding duplicates
        if aliases:
            current_aliases = set(map(str, entity.get("aliases", [])))
            for a in aliases:
                a = (a or "").strip()
                if a and a.lower() != name.lower():
                    current_aliases.add(a)
            entity["aliases"] = sorted(list(current_aliases))

        self._compiled_regex = None # Invalidate regex cache

    def _get_regex(self) -> re.Pattern:
        """Compiles and caches a regex to find all known entity names and aliases in text."""
        if self._compiled_regex is not None:
            return self._compiled_regex

        names: List[str] = []
        for base, meta in self.data.get("entities", {}).items():
            names.append(base)
            names.extend(meta.get("aliases", []))

        # Sort parts by length, longest first, to avoid partial matches (e.g., "Clary Fray" before "Clary")
        parts = sorted({re.escape(n) for n in names if n.strip()}, key=len, reverse=True)

        if not parts:
            # Return a regex that will never match anything
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

        # Map aliases back to their canonical names
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
    Extracts character/location descriptors from text using spaCy for robust
    linguistic analysis or a fallback regex method.
    """
    _VISUAL_KEYWORDS = {
        # Words that indicate potentially visual descriptors.  These keywords are used
        # to filter extracted clauses so that only phrases with imagery value are
        # returned.  When extending this set, prefer lower‑case singular forms to
        # maximize matching (e.g. "glass" would not match "glasses").
        "hair", "eye", "eyes", "beard", "freckle", "scar", "tattoo", "tall", "short",
        "slim", "thin", "stocky", "broad", "gaunt", "muscular", "built", "slender",
        "pale", "fair", "dark", "tan", "olive", "red", "blonde", "brown", "black",
        "blue", "green", "hazel", "grey", "gray", "golden", "face", "jaw", "cheek",
        "nose", "lips", "mouth", "soft", "sharp", "angular", "jacket", "coat",
        "dress", "boots", "jeans", "leather", "hoodie", "shirt", "gown", "cape",
        "young", "teenage", "fifteen", "old", "ancient", "slender", "lithe", "glasses",
        # Additional terms seen in the book that should trigger descriptor extraction
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
        """Suggests a list of descriptors found in the text."""
        if not text:
            return []

        # Use spaCy if available, otherwise fall back to simpler regex
        items = self._suggest_spacy(text) if self.nlp else self._suggest_regex(text)
        return self._dedupe(items)

    def _is_visual(self, s: str) -> bool:
        """Checks if a string contains any of our visual keywords."""
        return any(f' {k} ' in f' {s.lower()} ' for k in self._VISUAL_KEYWORDS)

    def _suggest_spacy(self, text: str) -> List[Descriptor]:
        """Extracts descriptors using spaCy's linguistic features."""
        out: List[Descriptor] = []
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ not in ["PERSON", "GPE", "LOC", "ORG"]:
                continue

            # --- Primary Description (e.g., "Clary is...") ---
            description = self._extract_spacy_description_for_entity(ent)
            if description and self._is_visual(description):
                 out.append(Descriptor(ent.text, description, self._default_aliases(ent.text)))

            # --- Possessive Descriptions (e.g., "The boy's eyes were green") ---
            # A Span's root token is the word that connects it to the rest of the sentence.
            # We look at this token's children for possessive relationships.
            root_token = ent.root
            for child in root_token.children:
                if child.dep_ == 'poss':
                    # child is the possessed noun (e.g., 'eyes')
                    possessed_noun = child
                    # Now find the description of this possessed noun
                    possessed_desc = self._extract_spacy_description_for_entity(possessed_noun)
                    if possessed_desc and self._is_visual(possessed_desc):
                        full_desc = f"{possessed_noun.text} {possessed_desc}"
                        out.append(Descriptor(ent.text, full_desc, self._default_aliases(ent.text)))
        return out

    def _extract_spacy_description_for_entity(self, ent_or_token: Span | Token) -> str:
        """
        Finds descriptive phrases linked to a spaCy entity or token.
        This handles cases like "Clary is tall" and "a girl with red hair".
        """
        # Pattern 1: Appositive (e.g., "Clary, a girl with red hair, ...")
        # We look for an appositional modifier directly following the entity.
        if hasattr(ent_or_token, 'rights'):
            for token in ent_or_token.rights:
                if token.dep_ == "appos":
                    # Reconstruct the full appositive phrase
                    appos_phrase = " ".join([t.text for t in token.subtree])
                    return appos_phrase.strip()

        # Pattern 2: Predicative Adjectives/Nouns (e.g., "Clary was tall", "Her hair was red")
        # Find the root verb of the entity/token and check if it's descriptive.
        # A Span object doesn't have a .head, but its .root token does.
        token_root = ent_or_token.root if isinstance(ent_or_token, Span) else ent_or_token

        verb_head = token_root.head
        if verb_head.lemma_ in ("be", "seem", "look", "appear"):
            # Collect attributes or adjectival complements
            full_desc_parts = []
            for child in verb_head.children:
                if child.dep_ in ("attr", "acomp"):
                    # Reconstruct the full descriptive phrase from the subtree
                    desc_phrase = " ".join(t.text for t in child.subtree)
                    full_desc_parts.append(desc_phrase)
            if full_desc_parts:
                return " ".join(full_desc_parts)

        return ""


    def _suggest_regex(self, text: str) -> List[Descriptor]:
        """A fallback method to extract descriptors using regular expressions."""
        out: List[Descriptor] = []
        last_person: Optional[str] = None

        # Regex for "Name, a/an/the <description>," or "Name is/was/had <description>."
        name_pattern = r"\b([A-Z][a-z']+(?:\s[A-Z][a-z']+)?)\b"
        patterns = [
            re.compile(rf"{name_pattern}\s*,\s*(?:a|an|the)?\s*([a-zA-Z\s\-]+?),", re.IGNORECASE),
            re.compile(rf"{name_pattern}\s+(?:is|was|had|wore)\s+([^\.]+)\.", re.IGNORECASE),
            re.compile(rf"(He|She)\s+(?:is|was|had|wore)\s+([^\.]+)\.", re.IGNORECASE),
            # NEW: “Name … in/with …”
            re.compile(rf"{name_pattern}[^\.]*?\b(?:in|with)\s+([^\,\.]+)", re.IGNORECASE),
            # NEW: “His/Her … hair …” (allows adjectives before hair)
            re.compile(r"(His|Her)[^\.]*?\bhair\s+(?:was|is|were|looked|appeared|seemed)?\s*([^\,\.]+)", re.IGNORECASE),
            # NEW: “His/Her glasses …”
            re.compile(r"(His|Her)\s+glasses\s+(?:were|are|looked|appeared|seemed|perched)?\s*([^\,\.]+)", re.IGNORECASE),
        ]

        for line in text.split('\n'):
            for pat in patterns:
                for match in pat.finditer(line):
                    name, desc = match.groups()
                    # Pronoun references ('he', 'she', 'his', 'her') are resolved to the
                    # most recently mentioned person (stored in last_person).  If the
                    # pronoun is found and we already have a last_person, we attach
                    # the description to them.  Otherwise, we treat the matched
                    # name as a new entity and store it in last_person for future
                    # pronoun resolution.
                    lower_name = name.lower()
                    # Only consider tokens that either represent a pronoun or begin with an
                    # uppercase letter.  This filters out accidental matches like
                    # 'scrubbed hair' which may be captured due to case‑insensitive
                    # matching of the regex.  A valid character name will always
                    # start with a capital letter.
                    is_pronoun = lower_name in ('he', 'she', 'his', 'her')
                    if not is_pronoun and not name[0].isupper():
                        continue  # skip lowercase names completely

                    if is_pronoun:
                        if last_person and self._is_visual(desc):
                            out.append(Descriptor(last_person, desc.strip(), []))
                    else:
                        if self._is_visual(desc):
                            out.append(Descriptor(name.strip(), desc.strip(), self._default_aliases(name.strip())))
                        last_person = name.strip()
        return out


    @staticmethod
    def _default_aliases(name: str) -> List[str]:
        """Generates simple aliases (like the first name) from a full name."""
        parts = name.split()
        if len(parts) > 1:
            return [parts[0]]
        return []

    @staticmethod
    def _dedupe(items: List[Descriptor]) -> List[Descriptor]:
        """Deduplicates a list of descriptors based on name and description."""
        merged: Dict[str, Descriptor] = {}
        for d in items:
            name_lower = d.name.lower()
            if name_lower not in merged:
                merged[name_lower] = d
            else:
                # Merge descriptions and aliases
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
    # Place context at the beginning for better prompt influence
    return f"In a scene where {snippet}: {base_prompt}" if snippet else base_prompt

