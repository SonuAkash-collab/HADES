# HADES L3 Gate — Cerberus Verification Layer

The Cerberus (Cerberus) gate guards the HADES memory hierarchy, ensuring only NLI-verified facts pass into permanent L3 storage.

## Technical Implementation

### 1. NLI Verification Pipeline
Verification is performed using a "Local-First" Natural Language Inference (NLI) model (`deberta-v3-base`).
*   **Claim Extraction**: We use `gliner-relex` to extract (Subject, Verb, Object) triples from the LLM's response.
*   **Localized Premise Construction**: For each claim, Cerberus searches the original document for the "Premise." It uses a keyword-overlap strategy (weighted by POS tags) to find the exact sentence in the source that *should* support the claim.
*   **Entailment Check**: The NLI model scores the relationship between the Premise and the Claim.
    *   **Entailment**: Claim is true based on source (Promoted to L3).
    *   **Contradiction**: Claim is false (Discarded).
    *   **Neutral**: Claim adds information not present in the source (Discarded).

### 2. Semantic CRC (Checksums)
To prevent "Bit Rot" or unauthorized modifications to the knowledge base, Cerberus implements a Semantic Cyclic Redundancy Check:
*   **Triple Hash**: Every verified triple is hashed using SHA-256.
*   **Integrity Check**: Before an L3 fact is used in future inference, its hash is re-verified. If the text has been tampered with or corrupted, the fact is invalidated.

---

## Benchmark Analysis
Note: For the finalized v1.0 global performance metrics across our multi-document evaluation suite, please refer to the [Main Repository README](../README.md).


## Running the Benchmark

```bash
python benchmarks/cerberus_apple_benchmark.py
```

Results are saved to `benchmarks/cerberus_benchmark_results.json`.
