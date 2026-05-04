"""Quick check: does the triple index contain the answers we need?"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pymupdf4llm
from shared.extractor import extract_source_triples
from cerberus.core import build_source_graph
from sentence_transformers import SentenceTransformer

STOP_MARKERS = ('## references', '## further reading', '## see also',
                '## external links', '## bibliography', '# references')

pdf_path = "Apple-1.pdf"
full_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
full_lower = full_text.lower()
for marker in STOP_MARKERS:
    idx = full_lower.find(marker)
    if idx != -1:
        full_text = full_text[:idx]
        break
full_text = re.sub(r'==> picture \[.*?\] intentionally omitted <==', '', full_text)
full_text = re.sub(r'\s+', ' ', full_text).strip()

triples = extract_source_triples(full_text)
all_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', full_text) if len(s.strip()) > 20]

embedder = SentenceTransformer('all-MiniLM-L6-v2')
source_graph = build_source_graph(triples, embedder=embedder, source_sentences=all_sentences)

print(f"Triple index size: {len(source_graph.triple_index)}")
print(f"Graph edges: {source_graph.graph.number_of_edges()}")

# Check for expected answers
targets = ["49%", "49", "10000", "10,000", "7500", "7,500", "amygdalin", "1625",
           "Snorri Sturluson", "Sturluson", "90.8", "Rosaceae", "Malus domestica",
           "Kazakhstan", "Malus sieversii", "Prose Edda", "Idunn"]

print("\n=== TRIPLE INDEX CONTENT SEARCH ===")
for target in targets:
    matches = [e["text"] for e in source_graph.triple_index if target.lower() in e["text"].lower()]
    if matches:
        print(f"  FOUND '{target}':")
        for m in matches[:3]:
            print(f"    -> {m}")
    else:
        print(f"  MISSING '{target}'")

# Also check if these facts exist in the raw source text
print("\n=== RAW PDF TEXT CHECK ===")
for target in targets:
    if target.lower() in full_text.lower():
        # Find surrounding context
        idx = full_text.lower().index(target.lower())
        context = full_text[max(0, idx-60):idx+60+len(target)]
        print(f"  IN PDF '{target}': ...{context}...")
    else:
        print(f"  NOT IN PDF '{target}'")
