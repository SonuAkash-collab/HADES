from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Iterable, Sequence

from sentence_transformers import CrossEncoder
from shared.triple import KnowledgeTriple


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """
    Loads the MS-MARCO Cross-Encoder model. 
    Using MiniLM-L-6-v2 for a balance of speed and accuracy on local hardware.
    """
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """
    Normalizes raw logits into a 0.0-1.0 probability range.
    """
    return 1 / (1 + np.exp(-x))


def verify_facts_against_query(
    query: str, 
    candidate_facts: Sequence[KnowledgeTriple], 
    threshold: float = 0.5
) -> list[tuple[KnowledgeTriple, float]]:
    """
    Strict Semantic Verification Gate.
    Uses a Cross-Encoder to verify which facts directly answer the user's query.
    
    Args:
        query: The user's natural language question.
        candidate_facts: A list of KnowledgeTriples retrieved from L1/L2/L3.
        threshold: Strictness of the filter (0.0 to 1.0).
        
    Returns:
        A list of (Triple, Score) tuples, sorted by relevance.
    """
    if not candidate_facts:
        return []

    model = _load_cross_encoder()
    
    # 1. Prepare pairs for cross-encoding
    # We use fact.as_text() to provide the full N-ary context (modality, condition, etc.)
    fact_texts = [f.as_text() for f in candidate_facts]
    pairs = [[query, text] for text in fact_texts]
    
    # 2. Get raw logits
    raw_scores = model.predict(pairs)
    
    # 3. Normalize scores using sigmoid
    normalized_scores = _sigmoid(raw_scores)
    
    # 4. Filter and pair with original triples
    verified_results: list[tuple[KnowledgeTriple, float]] = []
    
    # Handle single score vs multiple scores from model.predict
    if isinstance(normalized_scores, float):
        normalized_scores = [normalized_scores]
        
    for triple, score in zip(candidate_facts, normalized_scores):
        if score >= threshold:
            verified_results.append((triple, float(score)))
            
    # 5. Sort by descending relevance
    verified_results.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Top-K Slice (HADES Update: return max 10 facts to prevent context noise/conflicts)
    return verified_results[:10]
