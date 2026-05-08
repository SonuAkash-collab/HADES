import time
import streamlit as st
import app
from benchmarks.apple_pdf_benchmark import ingest_pdf
from shared.triple import KnowledgeTriple
from charon.core import rank_triples_by_importance, L1Cache

def stress_test_l2():
    print("Warming up...")
    app.get_embedder()
    app.get_cross_encoder()
    
    source_graph, _ = ingest_pdf("data/Apple-1.pdf")
    question = "Where does the apple tree originally come from?"
    st.session_state.source_graph = source_graph
    
    # 1. L1 Hit
    print("\n[MEASURING L1 HIT]")
    l1_cache = L1Cache(budgets={"facts": 400})
    # Pre-fill L1
    all_triples = []
    for u, v, d in source_graph.graph.edges(data=True):
        if 'triple' in d: all_triples.append(d['triple'])
        else: all_triples.append(KnowledgeTriple(subject=str(u), verb=str(d.get('verb', 'is')), object=str(v)))
    ranked = rank_triples_by_importance(all_triples)
    for triple, score in ranked:
        l1_cache.add_fact(triple, pagerank_score=score)
    
    st.session_state.l1_cache = l1_cache
    st.session_state.telemetry = {"memory_faults": [], "cerberus_log": [], "tool_calls": 0, "l1_status": "benchmark"}
    
    # Warm run
    app._chat_loop(question)
    
    start = time.time()
    app._chat_loop(question)
    l1_time = time.time() - start
    print(f"  L1 Time: {l1_time:.2f}s")

    # 2. L2 Stress Hit (Simulate a very large search space)
    print("\n[MEASURING L2 STRESS HIT (Page Fault)]")
    st.session_state.l1_cache = L1Cache(budgets={"facts": 400})
    
    # We will wrap the Cross-Encoder to simulate 1000 facts
    original_predict = app.get_cross_encoder().predict
    def mocked_predict(pairs):
        # Repeat the pairs until we have 1000
        stress_pairs = (pairs * (1000 // len(pairs) + 1))[:1000]
        return original_predict(stress_pairs)[:len(pairs)]
        
    app.get_cross_encoder().predict = mocked_predict
    
    start = time.time()
    app._chat_loop(question)
    l2_time = time.time() - start
    print(f"  L2 Stress Time: {l2_time:.2f}s")
    
    print(f"\nREALISTIC PAGE FAULT PENALTY: {l2_time - l1_time:.2f}s")

if __name__ == "__main__":
    stress_test_l2()
