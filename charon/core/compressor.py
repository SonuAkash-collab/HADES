from __future__ import annotations

import os
import re

from dotenv import load_dotenv
import ollama

from shared.triple import KnowledgeTriple

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


def compress_triples(triples: list[KnowledgeTriple], max_items: int = 5) -> str:
    selected = triples[:max_items]
    if not selected:
        return "No facts available to compress."
    return "\n".join(f"- {triple.as_text()}" for triple in selected)


def build_charon_prompt(triples: list[KnowledgeTriple]) -> str:
    fact_block = compress_triples(triples)
    return (
        "You are Charon, a local compression layer. "
        "Summarize the following facts into compact, factual prose without inventing new claims.\n"
        f"{fact_block}"
    )


def generate_charon_prose(triples: list[KnowledgeTriple]) -> str:
    prompt = build_charon_prompt(triples)
    messages = [
        {
            "role": "system",
            "content": "Return only compressed Charon-style text. No determiners. No filler.",
        },
        {"role": "user", "content": prompt},
    ]
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    content = response.get('message', {}).get('content', '').strip()
    return _enforce_charon_output(content)


def _enforce_charon_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^[\-*\d\.\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(
        r"\b(?:a|an|the)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    filler_patterns = [
        r"\b(?:here is|here are|i think|i can|let me|please note|sure)\b",
        r"\b(?:as an ai|in summary|to summarize)\b",
    ]
    for pattern in filler_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()
