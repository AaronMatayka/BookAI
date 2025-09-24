# src/adapters/ollama_api.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import requests

def ollama_extract_descriptors(
        model: str,
        text: str,
        url: str = "http://localhost:11434",
        max_len: int = 200,
) -> List[Tuple[str, str]]:
    prompt = f"""
You are a diligent assistant tasked with extracting visual descriptors
about characters from book text. For each named character in the
following passage, list a concise description of their appearance and
visual details. If none, either leave description empty after 'Name:' or use exactly:
no specific description

Format: one per line

Name: description

TEXT:
{text}
"""
    try:
        res = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        res.raise_for_status()
        data: Dict[str, Any] = res.json()
        response = (data.get("response") or "").strip()
    except Exception as e:
        print(f"[Ollama] Request failed: {e}")
        return []

    results: List[Tuple[str, str]] = []
    for line in response.splitlines():
        if ":" in line:
            name, desc = line.split(":", 1)
            name = name.strip()
            desc = desc.strip()
            if name and desc:
                results.append((name, desc))
    return results

def list_models(url: str = "http://localhost:11434") -> List[str]:
    endpoint = f"{url.rstrip('/')}/api/tags"
    names: List[str] = []
    try:
        res = requests.get(endpoint, timeout=10)
        res.raise_for_status()
        data: Dict[str, Any] = res.json()
        possible_lists = []
        if isinstance(data, dict):
            for key in ["models", "tags"]:
                lst = data.get(key)
                if isinstance(lst, list):
                    possible_lists.append(lst)
        for lst in possible_lists:
            for item in lst:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    for key in ["name", "model", "tag"]:
                        val = item.get(key)
                        if isinstance(val, str):
                            names.append(val)
                            break
    except Exception as e:
        print(f"[Ollama] Failed to fetch model list: {e}")
        return []
    seen = set()
    ordered: List[str] = []
    for n in names:
        if n not in seen:
            ordered.append(n)
            seen.add(n)
    return ordered

def ollama_generate_prompt(
        model: str,
        context: str,
        text: str,
        url: str = "http://localhost:11434",
) -> str:
    instruction = f"""
You are an expert at writing visual prompts for AI image generators.
Using CONTEXT and TEXT, write a single concise scene description that
weaves concrete visual details (characters, appearance, setting, action, mood).
One or two sentences. Do not include the words CONTEXT/TEXT.

CONTEXT:
{context}

TEXT:
{text}
""".strip()
    try:
        res = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": instruction, "stream": False},
            timeout=120,
        )
        res.raise_for_status()
        data: Dict[str, Any] = res.json()
        return (data.get("response") or "").strip()
    except Exception as e:
        print(f"[Ollama] Prompt generation failed: {e}")
        return ""
