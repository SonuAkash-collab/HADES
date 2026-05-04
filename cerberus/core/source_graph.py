from __future__ import annotations

import re
from dataclasses import dataclass, field
from hashlib import sha256

import networkx as nx
import numpy as np

from shared.triple import KnowledgeTriple


@dataclass(slots=True)
class SourceGraph:
    triples: list[KnowledgeTriple]
    graph: nx.DiGraph
    checksums: dict[str, str]
    master_checksum: str
    source_sentences: list[str] = None
    # L2 triple-content vector index: list of dicts with
    # {"text": str, "vector": np.ndarray, "subject": str, "verb": str, "object": str}
    triple_index: list[dict] = field(default_factory=list)


def triple_checksum(triple: KnowledgeTriple) -> str:
    return sha256(triple.as_text().encode("utf-8")).hexdigest()


def _build_triple_index(graph: nx.DiGraph, embedder, source_sentences: list[str] = None) -> list[dict]:
    """
    Build a vector index over full triple text AND source sentences.
    
    Two layers:
    1. Triple text (subject + verb + object) from graph edges
    2. Source sentences from the PDF — catches facts REBEL missed
       (e.g. numeric data like "49%", "10,000", "90.8 million tonnes")
    """
    if embedder is None:
        return []

    texts = []
    entries = []
    seen = set()

    # Layer 1: Graph edge triples
    for u, v, data in graph.edges(data=True):
        verb = str((data or {}).get("verb", "is")).strip()
        # Clean any residual <pad> tokens from REBEL output
        verb = re.sub(r'<pad>', '', verb).strip()
        triple_text = f"{u} {verb} {v}".strip()

        if triple_text in seen:
            continue
        seen.add(triple_text)

        texts.append(triple_text)
        entries.append({
            "text": triple_text,
            "vector": None,  # filled below via batched encode
            "subject": str(u),
            "verb": verb,
            "object": str(v),
        })

    # Layer 2: Source sentences (catches what REBEL missed)
    if source_sentences:
        for sent in source_sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 25 or sent_clean in seen:
                continue
            seen.add(sent_clean)
            texts.append(sent_clean)
            entries.append({
                "text": sent_clean,
                "vector": None,
                "subject": "",
                "verb": "",
                "object": "",
            })

    if not texts:
        return []

    # Batch encode all texts in a single call for efficiency
    vectors = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    for i, entry in enumerate(entries):
        entry["vector"] = vectors[i]

    return entries


def build_source_graph(triples: list[KnowledgeTriple], embedder=None, source_sentences: list[str] = None) -> SourceGraph:
    from charon.core import build_graph
    graph = build_graph(triples)
    checksums: dict[str, str] = {}

    for triple in triples:
        checksums[triple.as_text()] = triple_checksum(triple)

    # ── Entity coreference merging ──────────────────────────────
    # Collapse alias nodes into canonical entities before PageRank.
    # "Malus sieversii" and "wild ancestor" accumulate shared 
    # PageRank rather than splitting it across two nodes.
    # Uses batched embedding — ~80ms overhead for 100 nodes.
    if embedder is not None:
        from charon.core.graph import merge_similar_nodes
        graph, merged_count = merge_similar_nodes(
            graph, embedder, threshold=0.82
        )
        if merged_count > 0:
            import logging
            logging.info(
                f"[HADES] Entity merging: resolved {merged_count} "
                f"alias nodes → {graph.number_of_nodes()} canonical nodes"
            )

    # Embed nodes (still needed for node-level lookups elsewhere)
    for node in graph.nodes():
        if embedder:
            graph.nodes[node]["vector"] = embedder.encode(node)

    # ── Build triple-content vector index for L2 retrieval ──────
    triple_index = _build_triple_index(graph, embedder, source_sentences=source_sentences)

    master_text = "".join(sorted(triple.as_text() for triple in triples))
    master_checksum = sha256(master_text.encode("utf-8")).hexdigest()

    return SourceGraph(
        triples=triples,
        graph=graph,
        checksums=checksums,
        master_checksum=master_checksum,
        source_sentences=source_sentences,
        triple_index=triple_index,
    )
