from __future__ import annotations

from shared.l3_memory import fetch_clean_facts, initialize_l3_memory, save_fact
from shared.triple import KnowledgeTriple


def load_wiki() -> list[dict[str, str]]:
    initialize_l3_memory()
    records = fetch_clean_facts()
    return [
        {
            "subject": str(item.get("subject", "")).strip(),
            "verb": str(item.get("verb", "")).strip(),
            "object": str(item.get("object", "")).strip(),
        }
        for item in records
        if str(item.get("subject", "")).strip()
        and str(item.get("verb", "")).strip()
        and str(item.get("object", "")).strip()
    ]


def save_verified_fact(triple: KnowledgeTriple, source_page: int | None = None) -> None:
    save_fact(
        triple,
        source_page=source_page,
        cerberus_status="CLEAN",
    )
