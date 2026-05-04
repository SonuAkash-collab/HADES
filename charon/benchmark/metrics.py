from __future__ import annotations

import tiktoken


_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def sdpt(unique_acus_preserved: float, compressed_token_count: int) -> float:
    """
    Semantic Density per Token (SDpT).
    
    Returns tokens-per-ACU: the number of compressed tokens 
    required to represent each preserved Atomic Content Unit.
    
    Lower SDpT = better compression efficiency.
    A system that preserves 10 ACUs in 30 tokens (SDpT=3.0)
    is more efficient than one that uses 50 tokens (SDpT=5.0).
    
    Compare against raw_tokens / total_triples to get baseline SDpT.
    """
    if unique_acus_preserved <= 0:
        raise ValueError("unique_acus_preserved must be positive.")
    if compressed_token_count < 0:
        raise ValueError("compressed_token_count must be non-negative.")
    return compressed_token_count / unique_acus_preserved


def calculate_sdpt(unique_acus_preserved: float, compressed_token_count: int) -> float:
    return sdpt(unique_acus_preserved, compressed_token_count)
