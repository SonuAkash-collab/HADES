# HADES
### Hierarchical Adaptive Document Encoding System
*A local-first neural memory operating system for LLMs*

---

## The Problem

Context window bloat represents a significant inefficiency in modern LLM interaction. When users upload full documents to a model, they pay a high computational cost for every token, most of which are irrelevant to the specific query. This brute-force approach wastes the limited context window on filler text and formatting rather than dense, actionable information.

Information retrieval suffers from the Lost in the Middle phenomenon. Empirical evidence shows that LLMs lose accuracy when critical facts are buried deep within a long context block, often favoring information at the very beginning or end of a prompt. This architectural limitation makes it difficult for models to maintain reasoning consistency over lengthy technical documents or books.

Hallucination persistence creates a cycle of knowledge corruption. When an LLM generates an incorrect fact and that output is saved back into long-term memory without a verification step, the error becomes permanent. Traditional RAG systems lack a mechanism to audit the model's own contributions, allowing unverified claims to degrade the integrity of the persistent knowledge base over time.

---

## What HADES Does

HADES acts as an intelligent intermediary between raw documents and an LLM by managing a structured memory hierarchy. The system processes documents into a knowledge graph and serves only the most relevant fact cluster into the active context window to eliminate noise. An NLI verification gate audits every model output against the source graph before it can enter long-term memory, ensuring that only verified facts are stored. The entire pipeline runs locally on consumer hardware without external API dependencies or specialized GPU requirements.

```
PDF → Charon → Knowledge Graph (L2) → L1 Cache → LLM → Cerberus → L3
                   ↑                      ↑
              REBEL Triples          PageRank + MiniLM
```

---

## Architecture

### The Memory Hierarchy

| Tier | Name | Hardware Analogue | What it Does |
|------|------|-------------------|--------------|
| L1 | Elysium | CPU Cache | Active context window — 5 typed partitions with independent eviction policies |
| L2 | Asphodel | RAM | Full knowledge graph — PageRank-scored REBEL triples, MiniLM semantic search |
| L3 | Tartarus | Persistent Disk | Verified long-term memory — only NLI-verified facts written here |

### The Three Components

**Charon (Compression Pipeline)**
Charon manages the ingestion and transformation of raw data. It parses PDFs into markdown and extracts structured knowledge triples using REBEL, a seq2seq relation extraction model. These triples form a weighted NetworkX knowledge graph where nodes represent entities and edges represent relationships. At runtime, the system scores triples by PageRank centrality and reranks them using MiniLM cosine similarity to the user's specific query.

**Cerberus (NLI Verification Gate)**
Cerberus enforces a dirty-bit write-back policy to protect the integrity of the knowledge base. All generated model outputs initially enter a SCRATCH partition where they are marked as unverified. A DeBERTa-v3 cross-encoder runs Natural Language Inference against the original source graph to validate the claim. Facts that achieve an ENTAILMENT label are written to L3 storage, while those labeled as CONTRADICTION or NEUTRAL are discarded to prevent hallucinated data from reaching persistent memory.

**L1 Set-Associative Cache**
The L1 active context is divided into five typed sets to prevent context dilution. These include a pinned SYSTEM set, a FACTS set with PageRank-based eviction, a HISTORY set using Least Recently Used logic, a TOOLS set following First-In-First-Out priority, and a SCRATCH set that is fully flushed after verification. The token budget scales dynamically with document size: max(150, min(800, document_tokens ÷ 6)).

---

## Benchmark Results

### End-to-End QA (Apple Wikipedia PDF, 10 cases)
HADES achieves 90% accuracy on a 10-case factual QA benchmark over the Apple Wikipedia PDF (4,027 tokens), using only 12% of the document as active context at any given time. All 10 domains tested — botany, history, production, and culture — returned correct answers except one production statistic requiring cross-sentence numeric inference.

### Hallucination Detection (30 adversarial cases)
| Metric | Score |
|--------|-------|
| Accuracy | 81.8% |
| Precision | 92.3% |
| Recall | 70.6% |
| F1 | 0.80 |

All results produced using qwen2.5:1.5b running locally via Ollama — no external API calls, no GPU required, peak RAM approximately 3-4GB on a consumer laptop.

---

## Key Engineering Findings

1. Premise noise degrades NLI accuracy severely because DeBERTa-v3 achieves high confidence with a single clean premise sentence but drops to near-zero when given 3-5 competing sentences.
2. Open-world and closed-world extraction are different tasks that require REBEL for source graphs and spaCy SVO for claim fallback.
3. PageRank is near-uniform on graphs with fewer than 15 nodes which causes MiniLM query similarity to dominate fact selection for short documents.
4. Small models hallucinate rather than retrieve facts when given too much data so the fix is architectural: serve only the single most relevant triple to eliminate competing context.
5. REBEL num_return_sequences=3 provides 3x graph coverage at the same inference cost compared to single-sequence decoding.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| PDF Extraction | pymupdf4llm |
| Relation Extraction | REBEL (Babelscape/rebel-large) |
| Knowledge Graph | NetworkX + PageRank |
| Semantic Search | all-MiniLM-L6-v2 |
| NLI Verification | DeBERTa-v3-base (cross-encoder) |
| Local LLM | qwen2.5:1.5b via Ollama |
| Token Counting | tiktoken |
| Persistent Storage | SQLite WAL mode |
| UI | Streamlit |

---

## Quick Start

Prerequisites:
- Python 3.10 or 3.11 (not 3.12)
- Ollama installed from ollama.ai

```bash
git clone https://github.com/SiRex750/HADES.git
cd HADES
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:1.5b
```

Run:
```powershell
$env:PYTHONPATH="."; python -m streamlit run app.py
```

Note: First run downloads REBEL and DeBERTa weights (~600MB total). Subsequent runs use cached weights.

---

## Related Work

- MemGPT (2023) introduced the first operating system memory analogy for LLMs, and HADES adds graph compression and NLI write-back verification to this concept.
- LLMLingua (2023) focuses on token-level compression whereas HADES operates at the semantic triple level for higher precision.
- GraphRAG (2024) uses graph-based retrieval but HADES adds a critical verification layer that GraphRAG lacks.
- Lost in the Middle (2023) provided the empirical basis for the L1 partitioning strategy used in the HADES cache.

---

## Citation

```bibtex
@software{hades2026,
  author = {Siddanth Anil},
  title = {HADES: Hierarchical Adaptive Document Encoding System},
  year = {2026},
  institution = {PES University},
  url = {https://github.com/SiRex750/HADES}
}
```

---

## License
MIT
