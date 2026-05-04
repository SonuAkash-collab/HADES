
import sys
import os
sys.path.append(os.getcwd())
from benchmarks.apple_pdf_benchmark import ingest_pdf
from app import _chat_loop, _init_session_state
import streamlit as st

# Mock streamlit session state
if "l1_cache" not in st.session_state:
    _init_session_state()

source_graph = ingest_pdf("Apple-1.pdf")
st.session_state.source_graph = source_graph

query = "What percentage of global apple production does China account for in 2013?"
print(f"\nQuery: {query}")
answer = _chat_loop(query)
print(f"Answer: {answer}")

# Check telemetry
print("\nTelemetry:")
for fault in st.session_state.telemetry.get("memory_faults", []):
    print(f"- {fault}")
for log in st.session_state.telemetry.get("cerberus_log", []):
    print(f"- {log}")
