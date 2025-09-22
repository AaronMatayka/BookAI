#!/usr/bin/env python3
"""
Ollama Connector for BookAI

This module provides convenience functions for interacting with a locally
running Ollama server.  In addition to extracting visual descriptors, it
exposes helpers to list installed models and to generate concise visual
prompts suitable for AI image generators based on book text.

The default base URL for an Ollama server is ``http://localhost:11434``.
"""

import requests
from typing import List, Tuple, Dict, Any


def ollama_extract_descriptors(
        model: str,
        text: str,
        url: str = "http://localhost:11434",
        max_len: int = 200,
) -> List[Tuple[str, str]]:
    """
    Ask an Ollama model to extract named characters and their visual
    descriptors from a piece of text.

    Args:
        model: The Ollama model name (e.g. "phi3", "mistral", "llama3").
        text: The source text (book page).
        url: Base URL of the Ollama server.
        max_len: Unused currently but kept for API compatibility.

    Returns:
        A list of ``(name, description)`` tuples, where ``name`` is the
        character's name and ``description`` is a short description of
        their appearance or visual traits.
    """
    # Compose an instruction for the model.  We use a multi-line string
    # to improve readability.  The model is asked to list characters and
    # their visual descriptors on separate lines in a consistent format.
    prompt = f"""
You are a diligent assistant tasked with extracting visual descriptors
about characters from book text.  For each named character in the
following passage, list a concise description of their appearance and
visual details: clothing, hair, accessories, notable traits, etc. If you do not find any details on that character worth noting, either input nothing for said character, so list as
Name:

or input only exactly "no specific description" for their description.
Format your answer as one line per character using the form:

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
    """
    Query the Ollama server for a list of available models.

    The Ollama API exposes a ``/api/tags`` endpoint that returns a JSON
    payload describing all installed model tags.  This helper extracts
    the model names from that response.  If the API cannot be reached
    or returns an unexpected structure, an empty list is returned.

    Args:
        url: The base URL of the Ollama server.

    Returns:
        A list of model names (strings).  Duplicate names are removed
        while preserving order.
    """
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
    """
    Ask an Ollama model to craft a visual prompt for an AI image generator.

    The model receives a context snippet (e.g., previously learned
    descriptors) and the raw page text.  It is instructed to produce a
    concise visual description that an image generator can interpret.
    The prompt focuses on concrete visual details: characters, their
    appearance, setting, actions, mood, and optional style cues.

    Args:
        model: The Ollama model name (e.g. "llama3").
        context: A context string summarising known character details.
        text: The current page text.
        url: Base URL of the Ollama server.

    Returns:
        A single string containing the generated prompt.  If the API call
        fails, an empty string is returned.
    """
    instruction = f"""
You are an expert at writing visual prompts for AI image generators based on
literary text.  Using the provided CONTEXT and TEXT, write a single,
concise description of the scene focusing on concrete visual elements:

• Named characters and their appearances (hair, clothing, accessories).
  Always incorporate any known appearance details from the CONTEXT to
  enrich the character depictions in your description.
• Setting and environment (location, time of day, weather).
• Key actions or poses.
• Mood or atmosphere.
• Artistic style if relevant (e.g. oil painting, watercolor, anime).

Do not list characters individually; instead, weave the details into a
coherent, vivid sentence.  Do not include the words "CONTEXT" or "TEXT"
in your output, and do not repeat the raw text verbatim.  Limit your
description to one or two sentences.

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
        response = (data.get("response") or "").strip()
        return response
    except Exception as e:
        print(f"[Ollama] Prompt generation failed: {e}")
        return ""
