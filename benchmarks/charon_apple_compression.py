import time
import json
import re
from charon.core.cache import L1Cache
from shared.triple import KnowledgeTriple
from shared.extractor import extract_source_triples
from charon.core.graph import build_graph, rank_triples_by_importance
from charon.core.compressor import generate_charon_prose
from charon.benchmark.metrics import count_tokens, sdpt, calculate_sdpt
from app import get_embedder

SOURCE_TEXT = """
The apple tree comes from southern Kazakhstan, Kyrgyzstan, Uzbekistan, Turkey,
and northwestern part of China. Apples have been grown for thousands of years
in Asia and in Europe continent. They were brought to North America by European
World Colonial settlers. Apples have Religious and mythological significance in
many cultures.

Apples are generally grown by grafting, although wild apples grow readily from
seed. Apple trees are normally large if grown from seed, but small if grafted
onto roots (rootstock). There are more than 10,000 known variants of apples,
with a range of desired characteristics. Different variants are bred for various
tastes and uses: cooking, eating raw and cider production are the most common uses.

The seeds in apples can be fatal, but only if they have been crushed. Apples
contain amygdalin, which can release cyanide when digested. Apple seeds contain
a chemical that can make cyanide, but the amount is too small to harm people
who eat a few seeds.

In 2010, the fruit genome was sequenced as part of research on disease control
and selective breeding in apple production.

Worldwide production of apples in 2013 was 90.8 million tonnes. China grew
49% of the total.

The apple tree is a small, leaf-shedding tree that grows up to 3 to 12 metres
tall. The apple tree has a broad crown with thick twigs. The leaves are
alternately arranged simple ovals. They are 5 to 12 centimetres long and 3 to 6
centimetres wide. Blossoms come out in spring at the same time that the leaves
begin to bud. The flowers are white. They also have a slightly pink color. They
have five petals, and 2.5 to 3.5 centimetres in diameter. The fruit matures in
autumn. It is usually 5 to 9 centimetres in diameter. There are five carpels
arranged in a star in the middle of the fruit. Every carpel has one to three seeds.

The wild ancestor of apple trees is Malus sieversii. They grow wild in the
mountains of Central Asia in the north of Kazakhstan, Kyrgyzstan, Tajikistan,
and Xinjiang, China. Unlike domesticated apples, their leaves become red in
autumn. They are being used recently to develop Malus domestica to grow in
colder climates.

The apple tree was possibly the earliest tree to be cultivated. Its fruits have
become better over thousands of years. Alexander the Great of Greek civilisation
discovered dwarf apples in Asia Minor in 300 BC. Asia and Europe have used
winter apples as an important food for thousands of years.

Apples were brought to North America. The first apple orchard on the North
American continent was said to be near Boston in 1625. In the 1900s, costly
fruit industries, where the apple was a very important species, began developing.

In Norse mythology, the goddess Idunn gives apples to the gods in Prose Edda
written in the 13th century by Snorri Sturluson that makes them young forever.
English scholar H. R. Ellis Davidson suggests that apples were related to
religious practices in Germanic paganism.

The scientific name of the apple tree genus in the Latin language is Malus.
Most apples that people grow are of the Malus domestica species.

There are more than 7,500 known variants of apples. Different variants are
available for temperate and subtropical climates. One large collection of over
2,100 apple variants is at the National Fruit Collection in England.

Apples are grown around the world. China produces more than half of all
commercially grown apples. In 2020 and 2021, China produced 44,066,000 metric
tons. Other important producers were the European Union at 11,719,000 metric
tons, the United States at 4,490,000 metric tons, and Turkey at 4,300,000
metric tons. Total world production was 80,522,000 metric tons.

In the United Kingdom there are about 3000 different types of apples. The most
common apple type grown in England is the Bramley seedling, which is a popular
cooking apple.

Washington State currently produces over half of the Nation domestically grown
apples and has been the leading apple-growing State since the early 1920s.
New York and Michigan are the next two leading states in apple production.

Apples are in the group Maloideae. This is a subfamily of the family Rosaceae.
They are in the same subfamily as pears.
"""

def run_analysis():
    print("=" * 80)
    print("CHARON COMPRESSION ANALYSIS — Apple Wikipedia")
    print("=" * 80)

    raw_tokens = count_tokens(SOURCE_TEXT)
    print(f"      Source Tokens: {raw_tokens}")

    # 1. Extraction
    print("\n[1/4] Extracting triples from source...")
    source_triples = extract_source_triples(SOURCE_TEXT)
    total_triples = len(source_triples)
    print(f"      Extracted {total_triples} triples.")

    # 2. Ranking
    print("[2/4] Ranking triples (PageRank)...")
    ranked_triples = rank_triples_by_importance(source_triples)

    # 3. Cache Simulation (New 150 token budget)
    print("[3/4] Simulating Cache Hierarchy (150 token budget)...")
    
    embedder = get_embedder()
    queries = [
        "Where do apple trees come from?",
        "How big do apple trees grow?",
        "Who discovered dwarf apples?",
        "Is there cyanide in apple seeds?"
    ]

    results = []

    for query in queries:
        print(f"\n--- Query: {query} ---")
        
        # Fresh cache for each query to see re-ranking impact
        cache = L1Cache(budgets={"facts": 150})
        for triple, score in ranked_triples:
            cache.route_triple(triple, pagerank_score=score)
        
        # Re-rank for query
        cache.rerank_facts_for_query(query, embedder, alpha=0.3)
        
        # Final state (after re-ranking and potential eviction)
        final_triples = [e.triple for e in cache.active_facts.values()]
        
        compressed_text = generate_charon_prose(final_triples)
        
        tokens = count_tokens(compressed_text)
        acus = len(final_triples)
        score = sdpt(unique_acus_preserved=acus, compressed_token_count=tokens) if acus > 0 else 0.0
        
        print(f"      Tokens: {tokens} | ACUs: {acus} | SDpT: {score:.2f}")
        print(f"      Context Sample: {compressed_text[:100]}...")
        
        results.append({
            "query": query,
            "tokens": tokens,
            "acus": acus,
            "sdpt": score
        })

    # 4. Summary
    baseline_sdpt = raw_tokens / total_triples if total_triples > 0 else 0.0
    avg_sdpt = sum(r['sdpt'] for r in results) / len(results)
    print("\n" + "=" * 80)
    print("COMPRESSION SUMMARY")
    print("=" * 80)
    print(f"  Source Document Tokens                : {raw_tokens}")
    print(f"  Average SDpT across Apple Wiki queries: {avg_sdpt:.2f}")
    print(f"  Baseline SDpT (Raw Doc)               : {baseline_sdpt:.2f}")
    print(f"  Avg Improvement                       : {baseline_sdpt - avg_sdpt:.2f} tokens/ACU")
    print("=" * 80)

if __name__ == "__main__":
    run_analysis()
