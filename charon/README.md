# HADES L1 Cache — Charon Compression Layer

The Charon pipeline compresses raw documents into token-efficient, graph-ranked prose for passage into the HADES memory hierarchy. It functions as the **L1 Cache Controller**, managing the active context window.

## Technical Implementation

### 1. Importance Scoring (PageRank)
HADES uses graph theory to identify the most critical facts in a document. 
*   **Graph Construction**: Subject-Verb-Object triples are mapped as nodes (entities) and directed edges (relations).
*   **Centrality Analysis**: We apply the PageRank algorithm to the graph. Entities that are highly connected (e.g., "Apple Inc." in a corporate doc) receive higher scores.
*   **Triple Ranking**: A triple's importance score is the average PageRank of its subject and object. This ensures that facts connecting major entities are prioritized for the L1 cache.

### 2. Set-Associative Cache Routing
The L1 cache is divided into specialized partitions to maintain conversational coherence:
*   **SYSTEM**: Core instructions and constraints.
*   **FACTS**: Top-ranked knowledge triples from the document.
*   **HISTORY**: Previous user/assistant interactions (LRU eviction).
*   **TOOLS**: Available functions and API definitions.
*   **SCRATCH**: Temporary, unverified storage for current LLM output.

### 3. Hierarchical Recall (L2 Fallback)
If the L1 cache budget (e.g., 150 tokens) is too small to contain the answer, the system detects a "Page Fault" using a Cross-Encoder similarity check. It then "faults" to the L2 vector index to retrieve technical source sentences that were pruned during the initial compression.

---

## Benchmark Analysis: The 8-Case QA Suite

The Charon benchmark evaluates the system's ability to answer high-precision questions using only compressed context.

### Case Selection Strategy
*   **Technical (ATP/Mitochondria, CRISPR)**: Tests the system's ability to preserve complex scientific terminology.
*   **Historic (Caesar, Apollo 11)**: Tests proper noun preservation (Rubicon, Michael Collins).
*   **Numeric (NVIDIA Revenue, Interest Rates)**: Tests the most difficult dimension—preserving exact figures during prose condensation.
*   **Abstract (Transformers/Attention)**: Tests conceptual understanding.

### Result Analysis
| Metric | Score | Impact |
| :--- | :--- | :--- |
| **Accuracy** | **100.0%** | The combination of Top-3 REBEL extraction and Source Fallback ensures no "hallucination of omission." |
| **SDpT Improvement** | **+7.89** | Indicates that the compressed text contains 7.89 more facts per 100 tokens than the original raw text. |

**Significant Discovery**: The system achieved a **97.1% reduction** on the Apple Wikipedia article. This demonstrates that for large datasets, the PageRank filter effectively prunes "fluff" while the Hierarchical Recall ensures technical precision is maintained.

## Running the Benchmark

```bash
python benchmarks/charon_benchmark.py
```

Results are saved to `benchmarks/charon_benchmark_results.json`.
