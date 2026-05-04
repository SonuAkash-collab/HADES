
import sys
import os
sys.path.append(os.getcwd())
from cerberus.core.source_graph import build_source_graph
from cerberus.core.verifier import verify_claim
from shared.triple import KnowledgeTriple
from shared.extractor import extract_source_triples

SOURCE_TEXT = "Worldwide production of apples in 2013 was 90.8 million tonnes. China grew 49% of the total."
print("Extracting triples...")
triples = extract_source_triples(SOURCE_TEXT)
for t in triples:
    print(f"Triple: {t.as_text()}")

print("Building graph...")
graph = build_source_graph(triples)

claim = KnowledgeTriple(subject="China", verb="grew", object="49% of the total worldwide apple production")
print("Verifying claim...")
verdict = verify_claim(claim, graph, source_sentences=[SOURCE_TEXT])
print(f"Verdict: {verdict.label}")
print(f"Reason: {verdict.reason}")
