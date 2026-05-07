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

## Benchmark Analysis
Note: For the finalized v1.0 global performance metrics across our multi-document evaluation suite, please refer to the [Main Repository README](../README.md).


## Running the Benchmark

```bash
python benchmarks/charon_benchmark.py
```

Results are saved to `benchmarks/charon_benchmark_results.json`.
