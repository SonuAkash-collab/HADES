from __future__ import annotations

from functools import lru_cache
from typing import Iterable
import networkx as nx

from shared.triple import KnowledgeTriple


@lru_cache(maxsize=1)
def _load_spacy_sm():
    import spacy
    return spacy.load("en_core_web_sm")


def _is_valid_entity(label: str) -> bool:
    label = " ".join(label.split())  # normalise all whitespace
    """
    Validate entity labels using spaCy POS tagging.
    A valid entity must:
    - Contain at least one NOUN, PROPN, or NUM token
    - Not be longer than 40 characters
    - Not be a pure pronoun (I, it, they, he, she, we)
    - Not be a pure verb phrase
    """
    label = label.strip()
    
    # Hard length limits
    if len(label) > 100:
        return False
    if len(label) < 2:
        return False
    
    # Pure pronouns — never valid graph entities
    PRONOUNS = {
        "it", "It", "they", "They", "he", "He", "she", "She",
        "we", "We", "its", "Its", "their", "Their", "I", "them"
    }
    if label in PRONOUNS:
        return False
    
    # Contains citation artifacts from PDF extraction
    import re
    if re.search(r'\[\s*\d+\s*\]', label):  # [16], [2], etc.
        return False
    
    # Contains stray punctuation artifacts
    # Valid entities don't end with comma, open paren, or period+space
    if label.rstrip().endswith((',', '(', '. ')):
        return False
    if label.strip().startswith(('(', ')')):
        return False
    
    # Section headers from PDF — contain no relation to content
    SECTION_ARTIFACTS = {
        "Other websites", "Further reading", "References",
        "See also", "External links", "Notes", "Bibliography"
    }
    if label.strip() in SECTION_ARTIFACTS:
        return False
    
    # Starts with article + adjective pattern (not a proper entity)
    # "a little alcohol" "a broad crown" "a Green Delicious Apple"
    # These start with indefinite article
    lower = label.lower().strip()
    if lower.startswith(("a ", "an ")):
        return False
    
    # Always keep pure numeric values — dates, quantities, percentages
    # These are often filtered by prepositional phrase rules but are
    # critical factual entities
    numeric_only = re.sub(r'[\d\s\.\,\%]', '', label).strip()
    if not numeric_only:  # label is purely numeric/punctuation
        return True
    
    # Also keep labels that are short and contain a number
    # "49%", "1625", "90.8 million tonnes"
    if any(c.isdigit() for c in label) and len(label) <= 25:
        return True

    # Starts with preposition
    PREP_STARTS = (
        "of ", "for ", "into ", "from ", "with ", "by ",
        "to ", "in ", "on ", "at ", "as ",
        "simply ", "only ", "also ", "just ", "more ",
        "around ", "unlike ", "unlike ",
        "most of ", "many of ", "some of ", "all of ",
    )
    if any(lower.startswith(p) for p in PREP_STARTS):
        # Only reject if the label is short (likely a fragment)
        # If it's long, it's probably a descriptive fact object
        if len(label) < 20:
            return False
    
    # Citation template artifacts
    if "{" in label or "|" in label or "cite" in label.lower():
        return False
    
    # Must contain at least one noun, proper noun, or number
    try:
        nlp = _load_spacy_sm()
        doc = nlp(label)
        has_content = any(
            token.pos_ in {"NOUN", "PROPN", "NUM"}
            for token in doc
        )
        if not has_content:
            return False
        
        # Reject if first meaningful token is a verb
        first_meaningful = next(
            (t for t in doc if t.pos_ not in {"DET", "SPACE"}), 
            None
        )
        if first_meaningful and first_meaningful.pos_ == "VERB":
            return False
            
    except Exception:
        # If spaCy fails, fall back to length check only
        return len(label.split()) <= 6
    
    return True


def build_graph(triples: Iterable[KnowledgeTriple]) -> nx.MultiDiGraph:
    """
    Builds a MultiDiGraph from KnowledgeTriples, preserving N-ary properties.
    """
    from collections import Counter
    graph = nx.MultiDiGraph()
    triple_list = list(triples)
    
    # Aggregate identical triples (by all N-ary properties)
    freq: Counter = Counter(triple_list)
    unique_triples = set(triple_list)

    for triple in unique_triples:
        # Filter out artifact labels before adding to graph
        if not _is_valid_entity(triple.subject):
            continue
        if not _is_valid_entity(triple.object):
            continue
        
        if not graph.has_node(triple.subject):
            graph.add_node(triple.subject)
        if not graph.has_node(triple.object):
            graph.add_node(triple.object)
        
        weight = freq[triple]
        graph.add_edge(
            triple.subject,
            triple.object,
            verb=triple.verb,
            weight=weight,
            modality=triple.modality,
            is_negated=triple.is_negated,
            condition=triple.condition,
            temporal_anchors=triple.temporal_anchors,
            extraction_method=triple.extraction_method,
            is_deterministic=triple.is_deterministic
        )
    return graph


def pagerank_scores(graph: nx.MultiDiGraph) -> dict[str, float]:
    """
    Computes PageRank scores. Works with MultiDiGraph by summing weights of parallel edges.
    """
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, weight="weight")


def rank_triples_by_importance(triples: Iterable[KnowledgeTriple]) -> list[tuple[KnowledgeTriple, float]]:
    triple_list = list(triples)
    graph = build_graph(triple_list)
    scores = pagerank_scores(graph)
    
    # Compute initial scores
    ranked_with_scores = []
    for triple in triple_list:
        score = scores.get(triple.subject, 0.0) + scores.get(triple.object, 0.0)
        ranked_with_scores.append((triple, score))
    
    nlp = _load_spacy_sm()
    BOOST_CATEGORIES = {"PERSON", "GPE", "ORG", "DATE", "PERCENT", "QUANTITY"}

    # Batch encode all subjects and objects in one pass
    subjects = [triple.subject for triple, _ in ranked_with_scores]
    objects  = [triple.object  for triple, _ in ranked_with_scores]
    all_texts = subjects + objects
    all_docs  = list(nlp.pipe(all_texts, batch_size=64))
    subject_docs = all_docs[:len(subjects)]
    object_docs  = all_docs[len(subjects):]

    boosted_ranked = []
    for i, (triple, score) in enumerate(ranked_with_scores):
        sdoc = subject_docs[i]
        odoc = object_docs[i]
        has_entity_boost = False
        for doc in [sdoc, odoc]:
            if any(ent.label_ in BOOST_CATEGORIES for ent in doc.ents):
                score += 0.1
                has_entity_boost = True
                break
        if not has_entity_boost:
            for doc in [sdoc, odoc]:
                if any(t.pos_ == "PROPN" for t in doc):
                    score += 0.05
                    break
        boosted_ranked.append((triple, score))

    return sorted(boosted_ranked, key=lambda item: item[1], reverse=True)


def merge_similar_nodes(
    graph: nx.MultiDiGraph,
    embedder,
    threshold: float = 0.75,
) -> tuple[nx.MultiDiGraph, int]:
    """
    Merge graph nodes that refer to the same real-world entity.
    Redirects multi-edges while preserving their individual N-ary attributes.
    """
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return graph, 0

    # Batch encode ALL node labels in a single call for efficiency
    vectors = embedder.encode(
        nodes,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    sim_matrix = cos_sim(vectors)

    # Build merge map: alias → canonical (shorter label wins)
    merge_map: dict[str, str] = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if sim_matrix[i][j] >= threshold:
                node_i = merge_map.get(nodes[i], nodes[i])
                node_j = merge_map.get(nodes[j], nodes[j])
                if node_i == node_j:
                    continue
                # Canonical = shorter, more concise label
                if len(nodes[i]) <= len(nodes[j]):
                    merge_map[nodes[j]] = nodes[i]
                else:
                    merge_map[nodes[i]] = nodes[j]

    if not merge_map:
        return graph, 0

    # Build new MultiDiGraph redirecting edges to canonical nodes
    new_graph = nx.MultiDiGraph()

    for node in nodes:
        canonical = merge_map.get(node, node)
        if canonical not in new_graph:
            new_graph.add_node(canonical)

    for src, dst, data in graph.edges(data=True):
        new_src = merge_map.get(src, src)
        new_dst = merge_map.get(dst, dst)
        if new_src == new_dst:
            continue  # discard self-loops from merging
        
        # In a MultiDiGraph, we simply add the edge with its data.
        # This preserves distinct N-ary properties.
        new_graph.add_edge(new_src, new_dst, **data)

    return new_graph, len(merge_map)


def normalise_entities_with_llm(
    graph: nx.MultiDiGraph,
    ollama_model: str = "qwen3:0.6b",
) -> tuple[nx.MultiDiGraph, int]:
    """
    Optional second-pass entity normalisation using a local LLM.
    Redirects multi-edges while preserving their individual N-ary attributes.
    """
    import ollama
    import json

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return graph, 0

    # Only send short/ambiguous nodes to the LLM
    short_nodes = [n for n in nodes if len(str(n)) <= 25]
    long_nodes = [n for n in nodes if len(str(n)) > 25]

    if not short_nodes:
        return graph, 0

    prompt = (
        "You are an entity normalisation system for a knowledge graph.\n"
        "Given two lists of node labels, identify which SHORT nodes "
        "refer to the same entity as any LONG node.\n"
        "Output ONLY a valid JSON object mapping alias → canonical.\n"
        "If nothing maps, output {}.\n"
        "Do not invent mappings. Only map if you are highly confident.\n\n"
        f"SHORT nodes (possible aliases): {json.dumps(short_nodes)}\n"
        f"LONG nodes (possible canonicals): {json.dumps(long_nodes[:30])}\n\n"
        "Example output: "
        '{"it": "apple tree", "they": "apple trees", '
        '"the goddess": "Idunn"}'
    )

    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 300},
        )
        raw = response.get("message", {}).get("content", "{}").strip()
        # Strip markdown fences if present
        raw = raw.strip("```json").strip("```").strip()
        mapping = json.loads(raw)
        if not isinstance(mapping, dict):
            return graph, 0
    except Exception:
        return graph, 0

    # Apply the LLM mapping using the same merge logic
    merge_map = {
        str(alias): str(canonical)
        for alias, canonical in mapping.items()
        if str(alias) in nodes and str(canonical) in nodes
    }

    if not merge_map:
        return graph, 0

    new_graph = nx.MultiDiGraph()
    for node in nodes:
        canonical = merge_map.get(node, node)
        if canonical not in new_graph:
            new_graph.add_node(canonical)

    for src, dst, data in graph.edges(data=True):
        new_src = merge_map.get(src, src)
        new_dst = merge_map.get(dst, dst)
        if new_src == new_dst:
            continue
            
        new_graph.add_edge(new_src, new_dst, **data)

    return new_graph, len(merge_map)
