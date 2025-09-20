#!/usr/bin/env python3
"""
Ollama Connector for BookAI

This module provides convenience functions for interacting with a locally
running Ollama server. The API offered by Ollama exposes endpoints on
``/api``; the default base URL is ``http://localhost:11434``.  Here we
provide helper functions to extract character descriptors from text and to
enumerate the installed models on the server.  These routines are kept in
their own module so that the rest of the application can remain clean
and model-agnostic.
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
    Ask an Ollama model to extract descriptors from text.

    Args:
        model: The Ollama model name (e.g. "phi3", "mistral").
        text: The source text (book page).
        url: Base URL of the Ollama server.
        max_len: Max tokens/words in result.

    Returns:
        A list of (name, description) tuples.
    """
    prompt = f"""
    You are a helpful assistant designed to extract concrete visual details
    about characters from literature. For the given text you must list
    every named character and a succinct description of their visual or
    physical appearance. Include clothing, hair, accessories and any
    distinguishing traits. Format your response as one line per character
    using the following template:

    Name: description

    Only extract information that is explicitly present in the text. Do
    not invent details.

    TEXT:
    {text}
    """

    try:
        # Use the /api/generate endpoint to perform a single prompt completion.
        res = requests.post(
            f"{url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        res.raise_for_status()
        data: Dict[str, Any] = res.json()
        response = (data.get("response") or "").strip()
    except Exception as e:
        # If the request fails or times out, return an empty list so callers can fall back.
        print(f"[Ollama] Request failed: {e}")
        return []

    # Parse lines like "Simon: jeans and T-shirt...".  Split on the first colon.
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
    Queries the Ollama server for a list of available models.

    The Ollama API exposes a ``/api/tags`` endpoint that returns a JSON
    payload describing all installed model tags.  This helper attempts to
    extract the model names from that response.  If the API cannot be
    reached or returns an unexpected structure, an empty list is
    returned.

    Args:
        url: The base URL of the Ollama server.

    Returns:
        A list of model names (strings).  The order of the list follows
        the order provided by the server.  Duplicate names are removed.
    """
    endpoint = f"{url.rstrip('/')}/api/tags"
    names: List[str] = []
    try:
        res = requests.get(endpoint, timeout=10)
        res.raise_for_status()
        data: Dict[str, Any] = res.json()
        # The response may contain a 'models' list or 'tags' list depending
        # on the Ollama version.  Each item may be a string or a dict.
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
                    # Try common keys: 'name', 'model', 'tag'
                    for key in ["name", "model", "tag"]:
                        val = item.get(key)
                        if isinstance(val, str):
                            names.append(val)
                            break
    except Exception as e:
        print(f"[Ollama] Failed to fetch model list: {e}")
        return []
    # Deduplicate while preserving order
    seen = set()
    ordered_names: List[str] = []
    for n in names:
        if n not in seen:
            ordered_names.append(n)
            seen.add(n)
    return ordered_names
