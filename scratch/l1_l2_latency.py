import time
import streamlit as st
import app
from benchmarks.apple_pdf_benchmark import ingest_pdf
from shared.triple import KnowledgeTriple
from charon.core import rank_triples_by_importance, L1Cache

def time_l1_vs_l2():
    print("Warming up models...")
    app.get_embedder()
    app.get_cross_encoder()
    
    source_graph, _ = ingest_pdf("data/Apple-1.pdf")
    question = "Where does the apple tree originally come from?"
    st.session_state.source_graph = source_graph
    st.session_state.telemetry = {"memory_faults": [], "cerberus_log": [], "tool_calls": 0, "l1_status": "benchmark"}
    
    # 1. L1 Hit (Facts already in L1)
    print("\nSimulating L1 Hit (Hot Cache)...")
    all_triples = []
    for u, v, d in source_graph.graph.edges(data=True):
        if 'triple' in d: all_triples.append(d['triple'])
        else: all_triples.append(KnowledgeTriple(subject=str(u), verb=str(d.get('verb', 'is')), object=str(v)))
    
    ranked = rank_triples_by_importance(all_triples)
    l1_cache = L1Cache(budgets={"facts": 400})
    for triple, score in ranked:
        l1_cache.add_fact(triple, pagerank_score=score)
    
    st.session_state.l1_cache = l1_cache
    
    # Run once to ensure any remaining lazy loads are done
    app._chat_loop(question)
    
    start = time.time()
    _ = app._chat_loop(question)
    l1_time = time.time() - start
    print(f"  L1 Response Time: {l1_time:.2f}s")

    # 2. L2 Hit (Force a Page Fault by using empty L1)
    print("\nSimulating L2 Hit (Page Fault)...")
    st.session_state.l1_cache = L1Cache(budgets={"facts": 400}) # Empty but not None
    
    start = time.time()
    _ = app._chat_loop(question)
    l2_time = time.time() - start
    print(f"  L2 Response Time: {l2_time:.2f}s")
    
    print(f"\nPAGE FAULT PENALTY: {l2_time - l1_time:.2f}s")
    print(f"L1 is {l2_time / l1_time:.1f}x faster than L2.")

if __name__ == "__main__":
    time_l1_vs_l2()
