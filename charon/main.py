from __future__ import annotations

import os
import sys

from charon.core.cache import L1Cache
from charon.core.compressor import generate_charon_prose
from charon.core.graph import rank_triples_by_importance
from shared.extractor import extract_source_triples


def main(filename: str = "source_material.txt") -> None:
    default_text = (
        "Mitochondria generate ATP through oxidative phosphorylation in eukaryotic cells. "
        "Ribosomes synthesize proteins by translating messenger RNA sequences. "
        "The nucleus stores DNA and regulates gene expression through transcription factors. "
        "Lysosomes contain hydrolytic enzymes that digest macromolecules and cellular debris. "
        "The Golgi apparatus modifies proteins and lipids before packaging them into vesicles for transport."
    )

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as handle:
            source_text = handle.read()
    else:
        print("File not found, using default text.")
        source_text = default_text

    triples = extract_source_triples(source_text)
    ranked_triples = rank_triples_by_importance(triples)

    cache = L1Cache(budgets={"facts": 50})
    for triple, score in ranked_triples:
        cache.add_fact(triple, pagerank_score=score)

    active_triples = [entry.triple for entry in cache.active_facts.values()]
    final_text = generate_charon_prose(active_triples)

    print("(1) Total Triples Extracted")
    print(f"Count: {len(triples)}")
    print()

    print("(2) Triples Kept After Cache Eviction")
    print(f"Count: {len(active_triples)}")
    for idx, triple in enumerate(active_triples, start=1):
        print(f"{idx}. {triple.as_text()}")
    print()

    print("(3) Final Compressed Charon Text")
    print(final_text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "source_material.txt")
