import pymupdf4llm, re, json, os, datetime
import ollama
import streamlit as st
from shared.extractor import extract_source_triples
from charon.core.graph import build_graph, merge_similar_nodes
from charon.core import rank_triples_by_importance, L1Cache, generate_charon_prose
from cerberus.core import build_source_graph, verify_claim
from shared.triple import KnowledgeTriple
from sentence_transformers import SentenceTransformer
from charon.benchmark.metrics import count_tokens, sdpt as calculate_sdpt
from benchmarks.charon_benchmark import _check_accuracy, ask_judge
from typing import Callable, Sequence

# Pull in the main app dependencies and state
from app import _build_partitioned_messages, get_embedder, _chat_loop, OLLAMA_MODEL

import app
st.session_state = app._SESSION_STATE_FALLBACK

# Also patch it at the module level to be sure
from unittest.mock import patch
patcher = patch('streamlit.session_state', app._SESSION_STATE_FALLBACK)
patcher.start()
# ---------------------------------------------------------
import sys
def _safe_print(msg: str, **kwargs):
    """Safely print unicode to terminal by ignoring unencodable chars."""
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'ignore').decode('ascii'), **kwargs)
# ---------------------------------------------------------

STOP_MARKERS = (
    '## references', '## further reading', '## see also',
    '## external links', '## bibliography', '# references',
)

def ingest_pdf(pdf_path: str):
    """Ingests a PDF, caches triples, and returns the source graph and full text."""
    _safe_print(f"Ingesting {pdf_path}...")
    full_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
    
    # Clean up text
    full_lower = full_text.lower()
    for marker in STOP_MARKERS:
        idx = full_lower.find(marker)
        if idx != -1:
            full_text = full_text[:idx]
            break
            
    full_text = re.sub(r'==> picture \[.*?\] intentionally omitted <==', '', full_text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    # ---------------------------------------------------------
    # CACHING MECHANISM
    # ---------------------------------------------------------
    cache_dir = "benchmarks/cache"
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.basename(pdf_path)
    cache_path = os.path.join(cache_dir, f"{base_name}.json")
    
    if os.path.exists(cache_path):
        _safe_print(f"Loading cached triples from {cache_path}")
        with open(cache_path, "r") as f:
            data = json.load(f)
            triples = []
            for t in data["triples"]:
                # Convert list back to tuple for KnowledgeTriple
                if "temporal_anchors" in t and isinstance(t["temporal_anchors"], list):
                    t["temporal_anchors"] = tuple(t["temporal_anchors"])
                triples.append(KnowledgeTriple(**t))
    else:
        _safe_print(f"Extracting triples from {pdf_path} (this might take a while)")
        triples = extract_source_triples(full_text)
        from dataclasses import asdict
        with open(cache_path, "w") as f:
            json.dump({"triples": [asdict(t) for t in triples]}, f, indent=2)
            
    # Rebuild Graph
    all_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 20]
    source_graph = build_source_graph(triples, embedder=get_embedder(), source_sentences=all_sentences)
    
    _safe_print(f"Loaded {len(triples)} triples. Graph Nodes: {source_graph.graph.number_of_nodes()}")
    return source_graph, full_text

def run_naive_rag(full_text: str, query: str) -> str:
    """Run baseline naive RAG where the entire document text is stuffed in the prompt."""
    prompt = f"Document: {full_text}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided document text to answer the question."},
        {"role": "user", "content": prompt}
    ]
    model_name = st.session_state.get("selected_model", OLLAMA_MODEL)
    try:
        response = ollama.chat(model=model_name, messages=messages)
        return response.get('message', {}).get('content', '')
    except Exception as e:
        return f"Error running naive RAG: {e}"

def extract_hades_answer(raw_answer: str) -> str:
    """Extracts just the human-readable answer from HADES output, ignoring JSON CLAIMS block."""
    answer = re.sub(r'\n?CLAIMS:\s*\[.*?\]', '', raw_answer, flags=re.DOTALL).strip()
    if not answer or answer.startswith('CLAIMS:') or answer.startswith('{'):
        import json as _json
        claims_match = re.search(r'CLAIMS:\s*(\[.*?\])', raw_answer, re.DOTALL)
        if claims_match:
            try:
                claims = _json.loads(claims_match.group(1))
                if claims and isinstance(claims, list):
                    answer = claims[0].get('o', '') or claims[0].get('s', '')
            except Exception:
                pass
    return answer

def run_benchmarks():
    # ---------------------------------------------------------
    # 20-QUESTION BANK
    # ---------------------------------------------------------
    test_cases = [
        # Apple Domain (7)
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "Where does the apple tree originally come from?", "expected": "Kazakhstan"},
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "What percentage of global apple production does China account for in 2013?", "expected": "49%"},
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "What chemical in apple seeds can release cyanide?", "expected": "amygdalin"},
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "What was the total worldwide apple production in 2013?", "expected": "90.8 million tonnes"},
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "Who wrote the Prose Edda that mentions the goddess Idunn?", "expected": "Snorri Sturluson"},
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "When was the first apple orchard in North America established?", "expected": "1625"},
        {"domain": "Apple", "doc": "data/Apple-1.pdf", "query": "What is the scientific name of the cultivated apple species?", "expected": "Malus domestica"},

        # Nvidia Domain (7)
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "What was Nvidia's revenue in Q1 of FY25?", "expected": "$26 billion"},
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "How many employees does Nvidia have?", "expected": "31,000+"},
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "When was Nvidia founded?", "expected": "1993"},
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "How many developers are in the NVIDIA Developer Program?", "expected": "5 million"},
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "How many companies use NVIDIA AI technology to power AI factories?", "expected": "40,000"},
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "How many global startups are in NVIDIA Inception?", "expected": "19,000"},
        {"domain": "Nvidia", "doc": "data/corporate-nvidia-in-brief-pdf-august-3374577-FINAL.pdf", "query": "Who is the Founder and CEO of NVIDIA?", "expected": "Jensen Huang"},

        # NIPS Tech Domain (6)
        {"domain": "NIPS", "doc": "data/NIPS-2017-attention-is-all-you-need-Paper-1-4.pdf", "query": "What is the name of the new simple network architecture proposed in the paper?", "expected": "Transformer"},
        {"domain": "NIPS", "doc": "data/NIPS-2017-attention-is-all-you-need-Paper-1-4.pdf", "query": "What does the Transformer dispense with entirely?", "expected": "recurrence and convolutions"},
        {"domain": "NIPS", "doc": "data/NIPS-2017-attention-is-all-you-need-Paper-1-4.pdf", "query": "What is the BLEU score achieved by the model on the WMT 2014 English-to-German translation task?", "expected": "28.4"},
        {"domain": "NIPS", "doc": "data/NIPS-2017-attention-is-all-you-need-Paper-1-4.pdf", "query": "How many days was the model trained for on eight GPUs to establish a new single-model state-of-the-art BLEU score on the WMT 2014 English-to-French translation task?", "expected": "3.5 days"},
        {"domain": "NIPS", "doc": "data/NIPS-2017-attention-is-all-you-need-Paper-1-4.pdf", "query": "What attention mechanism outperforms dot product attention without scaling for larger values of dk?", "expected": "additive attention"},
        {"domain": "NIPS", "doc": "data/NIPS-2017-attention-is-all-you-need-Paper-1-4.pdf", "query": "What is used instead of performing a single attention function with dmodel-dimensional keys, values and queries?", "expected": "Multi-Head Attention"},
    ]

    # Group by document to optimize ingest cycles
    docs_to_test = {}
    for case in test_cases:
        docs_to_test.setdefault(case["doc"], []).append(case)

    results = []
    
    _safe_print("\n" + "="*100)
    _safe_print("HADES vs NAIVE RAG END-TO-END BENCHMARK")
    _safe_print("="*100)

    for doc_path, cases in docs_to_test.items():
        if not os.path.exists(doc_path):
            _safe_print(f"Error: {doc_path} not found. Ensure it exists in the root directory. Skipping...")
            continue
            
        source_graph, full_text = ingest_pdf(doc_path)
        
        # Hydrate the Session L1 Cache per Document
        all_triples = []
        for u, v, d in source_graph.graph.edges(data=True):
            if 'triple' in d:
                all_triples.append(d['triple'])
            else:
                all_triples.append(KnowledgeTriple(subject=str(u), verb=str(d.get('verb', 'is')), object=str(v)))
                
        ranked = rank_triples_by_importance(all_triples)
        l1_cache = L1Cache(budgets={"facts": 400})
        for triple, score in ranked:
            l1_cache.add_fact(triple, pagerank_score=score)
            
        st.session_state.l1_cache = l1_cache
        st.session_state.source_graph = source_graph

        for case in cases:
            _safe_print(f"\nProcessing [{case['domain']}] question: {case['query']}")
            
            # Reset telemetry
            st.session_state.telemetry = {
                "memory_faults": [], 
                "cerberus_log": [], 
                "tool_calls": 0, 
                "l1_status": "benchmark",
                "retrieved_triples": []  # <--- Add this
            }

            # 1. Run HADES Pipeline
            raw_hades_answer = _chat_loop(case["query"])
            hades_answer = extract_hades_answer(raw_hades_answer)
            hades_correct = _check_accuracy(hades_answer, case["expected"])
            hades_tokens = count_tokens(hades_answer)  # Minimal for HADES representation

            # --- NEW: Calculate Retrieval Hit ---
            retrieved_texts = [str(r) for r in st.session_state.telemetry.get("retrieved_triples", [])]
            retrieved_combined = " ".join(retrieved_texts).lower()
            retrieval_hit = case["expected"].lower() in retrieved_combined
            # ------------------------------------

            # 2. Run Naive RAG Baseline Pipeline
            naive_answer = run_naive_rag(full_text, case["query"])
            naive_correct = _check_accuracy(naive_answer, case["expected"])
            naive_tokens = count_tokens(full_text) + count_tokens(naive_answer)

            results.append({
                "query": case["query"],
                "expected": case["expected"],
                "hades_got": hades_answer,
                "hades_correct": hades_correct,
                "hades_tokens": hades_tokens,
                "retrieval_hit": retrieval_hit,  # <--- Add this
                "naive_got": naive_answer,
                "naive_correct": naive_correct,
                "naive_tokens": naive_tokens,
                "domain": case["domain"],
                "doc": doc_path
            })
            
            h_status = "PASS" if hades_correct else "FAIL"
            n_status = "PASS" if naive_correct else "FAIL"
            
            _safe_print(f"   Expected:   {case['expected']}")
            _safe_print(f"   HADES:      {h_status} | {hades_answer[:100]}... (Tokens: {hades_tokens})")
            if st.session_state.telemetry.get("retrieved_triples"):
                _safe_print(f"   Triples:    {st.session_state.telemetry['retrieved_triples']}")
            _safe_print(f"   Naive RAG:  {n_status} | {naive_answer[:100]}... (Tokens: {naive_tokens})")

    # ---------------------------------------------------------
    # FINAL REPORT GENERATION
    # ---------------------------------------------------------
    if not results:
        return

    hades_correct_count = sum(1 for r in results if r['hades_correct'])
    retrieval_hit_count = sum(1 for r in results if r['retrieval_hit']) # <--- Add this
    naive_correct_count = sum(1 for r in results if r['naive_correct'])
    total = len(results)
    
    hades_acc = (hades_correct_count / total) * 100
    retrieval_acc = (retrieval_hit_count / total) * 100 # <--- Add this
    naive_acc = (naive_correct_count / total) * 100
    
    _safe_print("\n" + "="*100)
    _safe_print("BENCHMARK SUMMARY REPORT")
    _safe_print("="*100)
    _safe_print(f"Retrieval Hit Rate: {retrieval_acc:.1f}%") # <--- Add this
    _safe_print(f"HADES Accuracy:     {hades_acc:.1f}%")
    _safe_print(f"Naive RAG Accuracy: {naive_acc:.1f}%")
    
    avg_hades_tokens = sum(r['hades_tokens'] for r in results) / total
    avg_naive_tokens = sum(r['naive_tokens'] for r in results) / total
    
    _safe_print(f"Avg Tokens (HADES): {avg_hades_tokens:.1f}")
    _safe_print(f"Avg Tokens (Naive): {avg_naive_tokens:.1f}")

    # Output to disk
    with open("benchmarks/apple_pdf_benchmark_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "hades_accuracy": hades_acc,
            "retrieval_hit_rate": retrieval_acc,
            "naive_rag_accuracy": naive_acc,
            "avg_hades_tokens": avg_hades_tokens,
            "avg_naive_tokens": avg_naive_tokens,
            "results": results
        }, f, indent=2)
    _safe_print("\n[SUCCESS] Detailed benchmark results saved to benchmarks/apple_pdf_benchmark_results.json")

if __name__ == "__main__":
    run_benchmarks()