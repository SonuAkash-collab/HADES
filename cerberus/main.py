from __future__ import annotations

from shared.extractor import extract_claim_triples, extract_source_triples
from cerberus.core.source_graph import build_source_graph
from cerberus.core.verifier import verify_claim
from cerberus.core.wiki_storage import save_verified_fact

def main() -> None:
    # 1. The Source Document (Ground Truth)
    source_text = (
        "The Saturn V rocket launched the Apollo 11 spacecraft from Kennedy Space Center. "
        "Neil Armstrong piloted the Lunar Module Eagle to a safe landing in the Sea of Tranquility. "
        "Buzz Aldrin deployed the solar wind composition experiment on the lunar surface."
    )

    # 2. Simulated AI Responses
    ai_response_true = "Neil Armstrong piloted the Lunar Module Eagle."
    ai_response_false = "Buzz Aldrin piloted the command module to Mars."

    print("--- CERBERUS VERIFICATION ORACLE ---\n")
    
    print("1. Extracting Source Graph (Semantic Checksum)...")
    source_triples = extract_source_triples(source_text)
    source_graph = build_source_graph(source_triples)
    print(f"   Nodes: {source_graph.graph.number_of_nodes()} | Edges: {source_graph.graph.number_of_edges()}")
    print(f"   Master Checksum (SHA-256): {source_graph.master_checksum[:16]}...\n")

    print("2. Testing Valid AI Response:")
    print(f"   AI Output: '{ai_response_true}'")
    true_claims = extract_claim_triples(ai_response_true)
    for claim in true_claims:
        print(f"   Claim Extracted: [{claim.as_text()}]")
        
        # Verify and Save
        result = verify_claim(claim, source_graph)
        status = "✅ VALID" if result.is_verified else "❌ INVALID"
        print(f"   Result: {status} ({result.reason})")
        
        if result.is_verified:
            save_verified_fact(claim)
            print("   -> Fact persisted to wiki.json")
    print()

    print("3. Testing Hallucinated AI Response:")
    print(f"   AI Output: '{ai_response_false}'")
    false_claims = extract_claim_triples(ai_response_false)
    for claim in false_claims:
        print(f"   Claim Extracted: [{claim.as_text()}]")
        
        # Verify and Block
        result = verify_claim(claim, source_graph)
        status = "✅ VALID" if result.is_verified else "❌ INVALID"
        print(f"   Result: {status} ({result.reason})")
        
        if not result.is_verified:
            print("   -> Blocked. Fact NOT saved to wiki.")

if __name__ == "__main__":
    main()