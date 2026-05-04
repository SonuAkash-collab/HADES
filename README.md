# HADES: Hierarchical Adaptive Document Encoding System

**The Neural Memory Management Unit (NMMU) for Verifiable AI.**

HADES is a local-first system designed to solve the two biggest problems in modern AI: **Context Window Saturation** (LLMs getting "overwhelmed" by long documents) and **Hallucination** (LLMs making things up).

---

## The Core Idea: The "OS" Analogy

Think of HADES as an **Operating System for an LLM's memory**. Instead of just dumping a whole PDF into an LLM, HADES manages information across three tiers, just like a computer handles data:

*   **L1 (CPU Cache) - The "Active Context"**: This is the tiny bit of text the LLM is actually reading right now. HADES uses a "Caveman" engine to compress 1,000 words into just ~30 words of pure fact, relevant *only* to your current question.
*   **L2 (RAM) - The "Knowledge Graph"**: This is a structured map of every fact in your document. If the L1 doesn't have the answer, HADES "faults" to the L2 to find the exact sentence or relation you need.
*   **L3 (SSD) - The "Verified Wiki"**: This is permanent storage. Only facts that have been "double-checked" by an NLI (Natural Language Inference) verifier are allowed to be saved here.

---

## Performance Evaluation & Benchmarks

HADES is evaluated across three distinct performance tiers to measure semantic density, factual recall, and verification precision.

### System Dashboard
| Suite | Component | Metric | Score | Engine |
| :--- | :--- | :--- | :--- | :--- |
| **CAVEMAN** | L1 Cache | **Factual Accuracy** | **100.0%** | REBEL + HADES L2 Fallback |
| **SENTINEL** | L3 Gate | **NLI Fidelity** | **81.8%** | DeBERTa-v3-base |
| **APPLE WIKI** | Compression | **Token Reduction** | **97.1%** | Qwen-2.5-1.5B (SLM) |

---

### 1. L1 Context Management (CAVEMAN)
The Caveman suite evaluates the system's ability to maintain high semantic density without losing query-critical information.

| Dimension | Result | Rationale |
| :--- | :--- | :--- |
| **Accuracy (8-case QA)** | **100.0%** | Achieved via **Hierarchical Recall** (triples + source sentences). |
| **Semantic Density (SDpT)** | **5.04** | Significant reduction in tokens required per Atomic Content Unit (ACU). |
| **L2 Fallback Rate** | ~12% | System successfully detects L1 cache misses and force-routes to L2 vectors. |

*Note: For long-form documents (>1k tokens), Caveman achieves a 35x compression ratio (1,000 → 29 tokens) while preserving 100% recall on technical entities. This is L1 context slicing; L2 storage reduction is approximately 5x.*

### 2. L3 Factual Fidelity (SENTINEL)
The Sentinel suite (Cerberus Gate) evaluates the NLI-driven write-back policy using 33 adversarial cases from the Apple-1 PDF.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Precision** | **85.2%** | Resistance to persisting model-generated hallucinations. |
| **Recall** | **79.3%** | Ability to verify complex numeric/temporal claims. |
| **F1 Score** | **0.82** | Harmonic mean of verification performance. |

---

## Architectural Innovations

The performance jumps recorded above are driven by three primary architectural breakthroughs implemented in the latest HADES release:

1.  **Hardware-Level Semantic Bypass**: To solve the "Confidence Bias" in Small Language Models (SLMs), HADES implements a Python-level arbitration layer. If the Cross-Encoder score for the current L1 context falls below **0.50**, the system forcefully routes the query to L2 memory, bypassing the LLM's potential "I don't know" or "Insufficient data" response.
2.  **Hybrid Hierarchical Recall**: Retrieval from L2 now utilizes a **Top-3 Joint Decoding** strategy. It returns both structured REBEL triples for symbolic reasoning and raw source sentences for high-precision numeric matching, ensuring that technical facts (like production percentages) are never lost during extraction.
3.  **Temporal & Numeric Anchoring**: Implementation of an NER-driven secondary scan that harvests temporal metadata structurally distant from the root verb. This ensures that quantitative metrics are always anchored to their correct timeframe, preventing temporal hallucinations.

---

## Installation & Setup

```bash
# 1. Setup Environment
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 2. Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Download Models
# Models (REBEL, DeBERTa, MS-MARCO) will download on first run (~800MB total).
```

### Quick Start
- **Web UI**: `streamlit run app.py`
- **Caveman Benchmark**: `python benchmarks/caveman_benchmark.py`
- **Sentinel Benchmark**: `python benchmarks/sentinel_apple_benchmark.py`

---

## Repository Structure

- `shared/` — Core `KnowledgeTriple` dataclass and REBEL-based extraction engine.
- `caveman/` — L1 Cache management: graph ranking, cache routing, and prose condensation.
- `sentinel/` — L3 Gate: NLI verification, source graph construction, and SQLite persistence.
- `benchmarks/` — Centralized directory for evaluation scripts and JSON results.
- `app.py` — Main Streamlit application and HADES orchestration logic.

---

## 📜 Academic Inspiration
HADES is inspired by research into Retrieval-Augmented Generation (RAG) and Hierarchical Memory Systems, including MemGPT, LLMLingua, and SummaC.
