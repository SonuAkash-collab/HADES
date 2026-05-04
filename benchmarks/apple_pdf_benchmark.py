import pymupdf4llm, re, json, os, datetime
from shared.extractor import extract_source_triples
from caveman.core.graph import build_graph, merge_similar_nodes
from caveman.core import rank_triples_by_importance, L1Cache, generate_caveman_prose
from sentinel.core import build_source_graph, verify_claim
from shared.triple import KnowledgeTriple
from sentence_transformers import SentenceTransformer
from caveman.benchmark.metrics import count_tokens, sdpt as calculate_sdpt
from caveman.benchmark.run_benchmark import _check_accuracy, ask_judge
from typing import Callable, Sequence
from app import _build_partitioned_messages, get_embedder
from caveman.core.semantic_arbitrator import verify_facts_against_query
import streamlit as st
from app import _chat_loop, get_embedder, _inject_clean_facts_into_l1

STOP_MARKERS = (
    '## references', '## further reading', '## see also',
    '## external links', '## bibliography', '# references',
)

def ingest_pdf(pdf_path: str):
    """Ingests a PDF and returns a build source graph."""
    print(f"Ingesting {pdf_path}...")
    # Use page_chunks=False to avoid cutting off sentences at page boundaries
    full_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
    
    # Clean up the text a bit
    full_lower = full_text.lower()
    for marker in STOP_MARKERS:
        idx = full_lower.find(marker)
        if idx != -1:
            full_text = full_text[:idx]
            break
            
    full_text = re.sub(r'==> picture \[.*?\] intentionally omitted <==', '', full_text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    # Extract triples
    triples = extract_source_triples(full_text)
    
    # Keep original sentences for verification
    all_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 20]
    
    # Use the official build_source_graph to ensure vectorization and metadata are correct
    source_graph = build_source_graph(triples, embedder=get_embedder(), source_sentences=all_sentences)
    
    # Report stats for logging
    print(f"Extracted {len(triples)} triples from full PDF.")
    print(f"Graph Nodes: {source_graph.graph.number_of_nodes()}")
    
    return source_graph

def run_apple_benchmark():
    pdf_path = "Apple-1.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    source_graph = ingest_pdf(pdf_path)
    
    # Extract initial L1 triples (most important)
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
    
    active_triples = [entry.triple for entry in l1_cache.active_facts.values()]

    # Initialise streamlit session state for _chat_loop
    if "l1_cache" not in st.session_state:
        st.session_state.l1_cache = l1_cache
    else:
        st.session_state.l1_cache = l1_cache

    if "source_graph" not in st.session_state:
        st.session_state.source_graph = source_graph
    else:
        st.session_state.source_graph = source_graph

    if "telemetry" not in st.session_state:
        st.session_state.telemetry = {
            "memory_faults": [],
            "sentinel_log": [],
            "tool_calls": 0,
            "l1_status": "benchmark"
        }

    test_cases = [
        {"domain": "botany", "query": "Where does the apple tree originally come from?", "expected": "Kazakhstan"},
        {"domain": "botany", "query": "What is the scientific name of the wild ancestor of apple trees?", "expected": "Malus sieversii"},
        {"domain": "production", "query": "What percentage of global apple production does China account for in 2013?", "expected": "49%"},
        {"domain": "production", "query": "How many known variants of apples are there?", "expected": "7500"},
        {"domain": "botany", "query": "What chemical in apple seeds can release cyanide?", "expected": "amygdalin"},
        {"domain": "history", "query": "When was the first apple orchard in North America established?", "expected": "1625"},
        {"domain": "culture", "query": "Who wrote the Prose Edda that mentions the goddess Idunn?", "expected": "Snorri Sturluson"},
        {"domain": "botany", "query": "What is the scientific name of the cultivated apple species?", "expected": "Malus domestica"},
        {"domain": "production", "query": "What was the total worldwide apple production in 2013?", "expected": "90.8 million tonnes"},
        {"domain": "botany", "query": "In which plant family are apples classified?", "expected": "Rosaceae"}
    ]

    results = []
    total_tokens = 0
    total_sdpt = 0.0

    print("\n" + "="*100)
    print("APPLE PDF END-TO-END BENCHMARK (with L2 Fallback)")
    print("="*100)

    for i, case in enumerate(test_cases, 1):
        print(f"\nProcessing [{case['domain']}] question: {case['query']}")
        
        start_time = datetime.datetime.now()
        
        # Reset telemetry per question
        st.session_state.telemetry["memory_faults"] = []
        st.session_state.telemetry["tool_calls"] = 0

        raw_answer = _chat_loop(case["query"])

        # Strip CLAIMS block — handle both newline and inline cases
        answer = re.sub(
            r'\n?CLAIMS:\s*\[.*?\]',
            '',
            raw_answer,
            flags=re.DOTALL
        ).strip()
        
        # Safety net: if stripped answer is empty or starts with JSON,
        # extract object value from raw_answer as fallback
        if not answer or answer.startswith('CLAIMS:') or answer.startswith('{'):
            import json as _json
            claims_match = re.search(r'CLAIMS:\s*(\[.*?\])', raw_answer, re.DOTALL)
            if claims_match:
                try:
                    claims = _json.loads(claims_match.group(1))
                    if claims and isinstance(claims, list):
                        # Extract the object field as the answer fallback
                        answer = claims[0].get('o', '') or claims[0].get('s', '')
                except Exception:
                    pass
        
        end_time = datetime.datetime.now()
        
        is_correct = _check_accuracy(answer, case["expected"])
        
        # Calculate tokens and compression
        prompt_tokens = count_tokens(answer) # Simplified for benchmark
        total_tokens += prompt_tokens
        
        results.append({
            "case": i,
            "query": case["query"],
            "expected": case["expected"],
            "got": answer,
            "correct": is_correct,
            "domain": case["domain"]
        })
        
        status = "PASS" if is_correct else "FAIL"
        print(f"Case {i:2}: {status} | {case['query']}")
        print(f"   Expected: {case['expected']}")
        print(f"   Got:      {answer[:100]}..." if len(answer) > 100 else f"   Got:      {answer}")

    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = (correct_count / len(test_cases)) * 100
    
    print("\n" + "="*100)
    print("APPLE PDF END-TO-END BENCHMARK REPORT (with L2 Fallback)")
    print("="*100)
    print(f"Overall Accuracy: {accuracy:.1f}%")
    
    # Domain breakdown
    domains = sorted(list(set(r['domain'] for r in results)))
    print("-" * 100)
    for d in domains:
        d_results = [r for r in results if r['domain'] == d]
        d_correct = sum(1 for r in d_results if r['correct'])
        d_acc = (d_correct / len(d_results)) * 100
        print(f"Domain [{d:<10}]: {d_correct}/{len(d_results)} ({d_acc:.1f}%)")
    print("-" * 100)

    for r in results:
        status = "PASS" if r['correct'] else "FAIL"
        print(f"Case {r['case']:2}: {status} | {r['query']}")
        print(f"   Expected: {r['expected']}")
        print(f"   Got:      {r['got']}")
        print("")

    # Save results
    with open("benchmarks/apple_pdf_benchmark_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)

    print(f"[SUCCESS] Detailed results saved to benchmarks/apple_pdf_benchmark_results.json")

if __name__ == "__main__":
    run_apple_benchmark()
