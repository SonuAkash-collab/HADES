# HADES L3 Gate — Sentinel Verification Layer

The Sentinel (Cerberus) gate guards the HADES memory hierarchy, ensuring only NLI-verified facts pass into permanent L3 storage.

## Technical Implementation

### 1. NLI Verification Pipeline
Verification is performed using a "Local-First" Natural Language Inference (NLI) model (`deberta-v3-base`).
*   **Claim Extraction**: We use `gliner-relex` to extract (Subject, Verb, Object) triples from the LLM's response.
*   **Localized Premise Construction**: For each claim, Sentinel searches the original document for the "Premise." It uses a keyword-overlap strategy (weighted by POS tags) to find the exact sentence in the source that *should* support the claim.
*   **Entailment Check**: The NLI model scores the relationship between the Premise and the Claim.
    *   **Entailment**: Claim is true based on source (Promoted to L3).
    *   **Contradiction**: Claim is false (Discarded).
    *   **Neutral**: Claim adds information not present in the source (Discarded).

### 2. Semantic CRC (Checksums)
To prevent "Bit Rot" or unauthorized modifications to the knowledge base, Sentinel implements a Semantic Cyclic Redundancy Check:
*   **Triple Hash**: Every verified triple is hashed using SHA-256.
*   **Integrity Check**: Before an L3 fact is used in future inference, its hash is re-verified. If the text has been tampered with or corrupted, the fact is invalidated.

---

## Benchmark Analysis: The Apple-1 PDF Suite

The Sentinel benchmark uses the official Apple-1 manual and history as a source of truth. It contains 33 "Adversarial" cases designed to trick the verifier.

### Difficulty Tiers
*   **Easy**: Direct matches (e.g., "Wozniak designed the Apple-1").
*   **Hard**: Paraphrased facts using synonyms or passive voice.
*   **Adversarial**: Claims that look plausible but have one incorrect number (e.g., "Apple-1 sold for $665" instead of $666.66). 
*   **Edge**: Multi-hop relations where the answer is split across two sentences.

### Result Analysis
| Metric | Score | Analysis |
| :--- | :--- | :--- |
| **Precision** | **85.2%** | Demonstrates high resistance to hallucinations. The verifier successfully caught nearly all adversarial numeric distortions. |
| **Recall** | **79.3%** | Indicates that ~20% of true facts were rejected. This is often due to "Neutral" scores on complex sentences where the NLI model is overly cautious. |

**Key Finding**: The verifier is tuned for **Safety Over Recall**. It is better to reject a true fact (Neutral) than to accept a false one (Entailment). This is a design requirement for the HADES "Verified Persistence" policy.

## Running the Benchmark

```bash
python benchmarks/sentinel_apple_benchmark.py
```

Results are saved to `benchmarks/sentinel_benchmark_results.json`.
