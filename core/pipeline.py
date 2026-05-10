from __future__ import annotations

import json
import os
import re
import tempfile
import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from charon.core import L1Cache, rank_triples_by_importance
from cerberus.core import build_source_graph, verify_claim
from shared.extractor import extract_claim_triples, extract_source_triples
from shared.l3_memory import fetch_clean_facts, save_fact
from shared.triple import KnowledgeTriple

# --- Constants ---
SYSTEM_INSTRUCTION = """You are the HADES NMMU. Answer questions using the provided Facts.

MODE 1 — CACHE HIT:
  Use this if the Facts contain the answer. Output ONLY plain text prose.
  DO NOT use any JSON, tool calls, or formatting like {"tool": ...} for a cache hit.
  Example:
  The apple tree originates from Kazakhstan.
  CLAIMS: [{"s": "apple tree", "v": "originates from", "o": "Kazakhstan"}]

MODE 2 — CACHE MISS:
  Use this if the Facts do not contain the answer. Output ONLY this JSON:
  {"tool": "search_memory", "keyword": "search_term"}

Rules:
1. Answer ONLY using the provided Facts.
2. If Facts are insufficient, you MUST use MODE 2.
3. Prose answer must come BEFORE the CLAIMS line.
4. NEVER repeat the user's question. If you don't know, use MODE 2.
5. Combine information from multiple facts if necessary to deduce the answer.
6. If still no answer after searching, say 'INSUFFICIENT DATA'.
7. CRITICAL: NEVER output '{"tool": "cache_hit"...}'. That tool does not exist. If you have the answer, just say it as plain text."""

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
OLLAMA_OPTIONS = {
    "temperature": 0.0,
    "num_predict": 512,
    "repeat_penalty": 1.05,
}

# --- Pure Logic Helpers ---

def safe_print(msg: str, **kwargs):
    """Safely print unicode to terminal by ignoring unencodable chars."""
    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'ignore').decode('ascii'), **kwargs)

def required_system_budget() -> int:
    return max(128, len(SYSTEM_INSTRUCTION.split()) * 3)

def push_telemetry_item(state: dict, key: str, value: str, max_items: int = 12) -> None:
    items = state["telemetry"].setdefault(key, [])
    items.append(value)
    if len(items) > max_items:
        del items[:-max_items]

def triple_key(triple: KnowledgeTriple) -> tuple[str, str, str]:
    return (
        str(triple.subject).strip().lower(),
        str(triple.verb).strip().lower(),
        str(triple.object).strip().lower(),
    )

def resolve_source_page(
    triple: KnowledgeTriple,
    source_page_lookup: dict[tuple[str, str, str], int],
) -> int:
    key = triple_key(triple)
    if key in source_page_lookup:
        return source_page_lookup[key]

    subj, verb, obj = key
    for (src_subj, src_verb, src_obj), page in source_page_lookup.items():
        if src_subj == subj and src_obj == obj:
            return page
        if src_subj == subj and src_verb == verb:
            return page

    return 0

def inject_clean_facts_into_l1(cache: L1Cache, limit: int = 64) -> int:
    injected = 0
    for fact in fetch_clean_facts()[-limit:]:
        subject = str(fact.get("subject", "")).strip()
        verb = str(fact.get("verb", "")).strip()
        object_text = str(fact.get("object", "")).strip()
        if not (subject and verb and object_text):
            continue

        triple = KnowledgeTriple(subject, verb, object_text)
        if triple.as_text() in cache.set_facts:
            continue

        cache.add_fact(triple, pagerank_score=-1.0)
        injected += 1

    return injected

def extract_search_keyword(content: str) -> str | None:
    match = re.search(r'(\{.*?"tool"\s*:\s*"search_memory".*?\})', content, re.DOTALL)
    if not match:
        return None
    
    try:
        parsed = json.loads(match.group(1))
        if not isinstance(parsed, dict) or parsed.get("tool") != "search_memory":
            return None
        keyword = str(parsed.get("keyword", "")).strip()
        if not keyword:
            return None
        
        simplifications = [
            (r'^(the|a|an)\s+', ''),
            (r'\s+(scientific name|common name)\s*', ' '),
            (r'^(origin|source|location)\s+of\s+', ''),
            (r'^(author|writer|creator)\s+of\s+', ''),
            (r'\s+mentioning\s+.*$', ''),
        ]
        simplified = keyword.lower().strip()
        for pattern, replacement in simplifications:
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE).strip()
        
        if simplified and len(simplified) < len(keyword) * 0.75:
            return simplified
        return keyword
    except json.JSONDecodeError:
        return None

def build_partitioned_messages(cache: L1Cache, prompt: str, forced_facts: list[str] = None) -> list[dict[str, str]]:
    facts = forced_facts if forced_facts is not None else [entry.text for entry in cache.set_facts.values()]
    
    numbers_map = {}
    collisions = []
    
    for f in facts:
        nums = re.findall(r'\b\d+(?:[\.,]\d+)*\b', f)
        if nums:
            key = re.sub(r'\b\d+(?:[\.,]\d+)*\b', 'NUM', f).strip().lower()
            if key not in numbers_map:
                numbers_map[key] = []
            numbers_map[key].append(f)
            
    for key, related_facts in numbers_map.items():
        if len(related_facts) > 1:
            all_nums = [set(re.findall(r'\b\d+(?:[\.,]\d+)*\b', rf)) for rf in related_facts]
            if any(n != all_nums[0] for n in all_nums):
                collisions.append(related_facts)

    facts_block = ""
    if collisions:
        facts_block += "<SYSTEM_WARNING: CONFLICTING DATA>\n"
        for group in collisions:
            for cf in group:
                facts_block += f"- {cf}\n"
        facts_block += "Caution: Multiple numerical counts exist. Resolve based on the precise wording of the query.\n"
        facts_block += "</SYSTEM_WARNING>\n\n"
    
    collided_set = {f for group in collisions for f in group}
    remaining_facts = [f for f in facts if f not in collided_set]
    facts_block += "\n".join(f"- {fact}" for fact in remaining_facts) if remaining_facts else ""
    if not facts_block:
        facts_block = "<empty>"

    user_turns = [turn.text for turn in cache.set_history if turn.role == "user"]
    last_two_user_turns = user_turns[-2:]
    user_turns_block = "\n".join(f"- {turn}" for turn in last_two_user_turns) if last_two_user_turns else "<empty>"

    user_content = (
        "L1 Context (Facts):\n"
        f"{facts_block}\n\n"
        "Last 2 User Messages:\n"
        f"{user_turns_block}\n\n"
        f"Current Question: {prompt}"
    )

    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]

def call_policy_model(messages: list[dict[str, str]], model_name: str) -> str:
    request_messages = [dict(message) for message in messages]
    
    if "qwen3" in model_name:
        for msg in request_messages:
            if msg["role"] == "system":
                msg["content"] += "\nCRITICAL INSTRUCTION: DO NOT output <think> tags. Do not explain your reasoning. Output only the final formatted answer immediately."
                break
    
    request_messages.append({"role": "assistant", "content": ""})

    response = ollama.chat(
        model=model_name,
        messages=request_messages,
        options=OLLAMA_OPTIONS,
    )
    content = response.get("message", {}).get("content", "").strip()
    
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content = re.sub(r"^(MODE\s*\d.*?CACHE\s*HIT\s*:\s*|MODE\s*\d.*?:\s*|CACHE\s*(HIT|MISS).*?:\s*)", "", content, flags=re.IGNORECASE).strip()
    return content

# --- Core Pipeline Orchestration ---

def query_l2_memory(query: str, keyword: str, source_graph, embedder, cross_encoder, is_benchmark: bool = False) -> str:
    if source_graph is None or not keyword:
        return ""

    triple_index = getattr(source_graph, 'triple_index', None)
    if not triple_index:
        return ""

    if is_benchmark:
        safe_print(f"   [L2 Memory] Encoding query vector...")
    query_vector = embedder.encode(query)

    scored_triples = []
    for entry in triple_index:
        if entry["vector"] is not None:
            sim = cosine_similarity([query_vector], [entry["vector"]])[0][0]
            scored_triples.append((entry["text"], sim))

    if is_benchmark:
        safe_print(f"   [L2 Memory] Stage 1 (Bi-Encoder) complete. Scored {len(scored_triples)} triples.")

    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_triples[:15]

    if not top_candidates or top_candidates[0][1] < 0.25:
        return ""

    candidate_facts = [text for text, _ in top_candidates]
    pairs = [[query, fact] for fact in candidate_facts]
    
    if is_benchmark:
        safe_print(f"   [L2 Memory] Stage 2 (Cross-Encoder) reranking {len(pairs)} pairs...", flush=True)
    
    scores = cross_encoder.predict(pairs)
    
    scored_candidates = sorted(zip(candidate_facts, scores), key=lambda x: x[1], reverse=True)
    top_facts = [fact for fact, score in scored_candidates[:3]]
    
    if is_benchmark:
        safe_print(f"   [L2 Memory Hit] Found {len(top_facts)} facts for keyword '{keyword}'")
        for f in top_facts:
            safe_print(f"     - {f}")
    return "\n".join(top_facts)

def query_l3_wiki(keyword: str, embedder) -> str:
    from shared.l3_memory import fetch_clean_facts_by_similarity
    results = fetch_clean_facts_by_similarity(keyword=keyword, embedder=embedder, threshold=0.55, limit=3)

    if not results:
        return ""

    lines = []
    for fact in results:
        triple_text = f"{fact['subject']} {fact['verb']} {fact['object']}".strip()
        source_page = int(fact.get("source_page") or 0)
        citation = f" (source_page={source_page})" if source_page > 0 else ""
        lines.append(f"{triple_text}{citation}")

    return " | ".join(lines)

def process_pdf(file_path: str, state: dict, embedder) -> tuple[int, int]:
    import pymupdf4llm
    
    md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)

    triples: list[KnowledgeTriple] = []
    source_page_lookup: dict[tuple[str, str, str], int] = {}
    all_sentences: list[str] = []

    STOP_MARKERS = ("## references", "## further reading", "## see also", "## external links", "## bibliography", "## other websites", "# references", "# further reading")

    for page_number, page_chunk in enumerate(md_text, start=1):
        page_text = page_chunk.get("text", "")
        page_lower = page_text.lower()
        if any(marker in page_lower for marker in STOP_MARKERS):
            for marker in STOP_MARKERS:
                idx = page_lower.find(marker)
                if idx != -1:
                    page_text = page_text[:idx]
                    break

        page_text = re.sub(r'==> picture \[.*?\] intentionally omitted <==', '', page_text)
        raw_markdown = page_text
        clean_text = re.sub(r'^#{1,6}\s+.*$', '', page_text, flags=re.MULTILINE)
        clean_text = re.sub(r'\*\*|__|\*|_', '', clean_text)
        clean_text = re.sub(r'\[\s*\d+\s*\]', '', clean_text)
        clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if not clean_text or len(clean_text) < 30:
            continue

        page_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 20]
        all_sentences.extend(page_sentences)

        page_triples = extract_source_triples(raw_markdown)
        triples.extend(page_triples)

        for triple in page_triples:
            source_page_lookup.setdefault(triple_key(triple), page_number)

    if not triples:
        state["source_graph"] = None
        state["triple_source_pages"] = {}
        return 0, 0
    
    source_graph = build_source_graph(triples, embedder=embedder, source_sentences=all_sentences)
    state["source_graph"] = source_graph
    state["triple_source_pages"] = source_page_lookup

    cache: L1Cache = state["l1_cache"]
    cache.set_facts.clear()
    cache.set_tools.clear()

    ranked = rank_triples_by_importance(triples)
    

    for triple, score in ranked:
        cache.add_fact(triple, pagerank_score=score)

    injected_from_l3 = inject_clean_facts_into_l1(cache)
    state["telemetry"]["l1_status"] = "pdf_loaded"

    if state.get("deep_entity_resolution", False):
        from charon.core.graph import normalise_entities_with_llm
        source_graph.graph, extra_merges = normalise_entities_with_llm(
            source_graph.graph,
            ollama_model=state.get("selected_model", OLLAMA_MODEL),
        )
        if extra_merges > 0:
            push_telemetry_item(state, "memory_faults", f"Deep resolution: {extra_merges} additional aliases resolved")

    nodes_after = source_graph.graph.number_of_nodes()
    push_telemetry_item(state, "memory_faults", (f"PDF ingested: {len(triples)} raw triples → {nodes_after} merged graph nodes (L3 injected: {injected_from_l3})"))
    return len(triples), nodes_after

def run_cerberus_writeback(final_answer: str, state: dict) -> bool:
    source_graph = state.get("source_graph")
    if source_graph is None:
        push_telemetry_item(state, "cerberus_log", "No source graph loaded; Cerberus verification skipped.")
        return True

    answer_triples = []
    claims_match = re.search(r'CLAIMS:\s*(\[.*?\])', final_answer, re.DOTALL)
    if claims_match:
        try:
            raw_claims = json.loads(claims_match.group(1))
            for c in raw_claims:
                if isinstance(c, dict) and c.get("s") and c.get("v") and c.get("o"):
                    answer_triples.append(KnowledgeTriple(
                        subject=str(c["s"]).strip(),
                        verb=str(c["v"]).strip(),
                        object=str(c["o"]).strip(),
                        extraction_method="llm_structured",
                        is_deterministic=True,
                    ))
        except (json.JSONDecodeError, TypeError):
            pass
    
    if not answer_triples:
        from shared.extractor import _extract_svo_triples
        answer_triples = _extract_svo_triples(final_answer)
    
    if not answer_triples:
        push_telemetry_item(state, "cerberus_log", "No triples extracted from assistant answer.")
        return True

    source_page_lookup = state.get("triple_source_pages", {})
    has_contradiction = False

    for triple in answer_triples:
        verdict = verify_claim(triple, source_graph, source_sentences=source_graph.source_sentences)
        if verdict.is_verified:
            source_page = resolve_source_page(triple, source_page_lookup)
            inserted = save_fact(triple, source_page=source_page, cerberus_status="CLEAN")
            citation = f"p.{source_page}" if source_page > 0 else "p.?"
            write_result = "stored" if inserted else "duplicate_ignored"
            push_telemetry_item(state, "cerberus_log", f"✅ CLEAN | {triple.as_text()} | {verdict.reason} | {citation} | {write_result}")
        elif verdict.label == "contradiction":
            has_contradiction = True
            push_telemetry_item(state, "cerberus_log", f"❌ CONTRADICTION | {triple.as_text()} | {verdict.reason}")
        else:
            push_telemetry_item(state, "cerberus_log", f"⚠️ NEUTRAL | {triple.as_text()} | {verdict.reason}")

    inject_clean_facts_into_l1(state["l1_cache"])
    return not has_contradiction

def chat_loop(prompt: str, state: dict, embedder, cross_encoder) -> str:
    is_benchmark = state["telemetry"].get("l1_status") == "benchmark"
    if is_benchmark:
        safe_print(f"\n   [HADES] Processing query: '{prompt}'", flush=True)
    
    cache: L1Cache = state["l1_cache"]
    source_graph = state.get("source_graph")

    inject_clean_facts_into_l1(cache)
    cache.add_history_turn("user", prompt)

    active_facts = [entry.text for entry in cache.set_facts.values()]
    l2_result = query_l2_memory(prompt, prompt, source_graph, embedder, cross_encoder, is_benchmark)
    
    if is_benchmark:
        safe_print(f"   [HADES] L2 retrieval complete. Merging with L1...", flush=True)
    
    l2_facts = l2_result.split("\n") if l2_result else []

    if not active_facts and not l2_facts:
        forced_facts = []
    elif not active_facts:
        forced_facts = l2_facts
    else:
        combined = list(set(active_facts + l2_facts))
        pairs = [[prompt, f] for f in combined]
        if is_benchmark:
            safe_print(f"   [Cross-Encoder] Reranking {len(combined)} candidate facts...")
        scores = cross_encoder.predict(pairs)
        if is_benchmark:
            safe_print(f"   [Cross-Encoder] Reranking complete.")
        scored_facts = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
        forced_facts = [f for f, s in scored_facts[:5]]

    if is_benchmark:
        state["telemetry"].setdefault("retrieved_triples", []).extend(forced_facts)
        safe_print(f"   [L1 Retrieval] Handed {len(forced_facts)} facts to LLM:", flush=True)
        for f in forced_facts:
            safe_print(f"     - {f}", flush=True)

    conversation = build_partitioned_messages(cache, prompt, forced_facts=forced_facts)
    content = call_policy_model(conversation, state.get("selected_model", OLLAMA_MODEL))
    
    keyword = extract_search_keyword(content)
    if keyword:
        already_searched = any(
            f"keyword='{keyword}'" in item
            for item in state["telemetry"].get("memory_faults", [])
            if isinstance(item, str)
        )
        
        if not already_searched:
            state["telemetry"]["tool_calls"] = state["telemetry"].get("tool_calls", 0) + 1
            l2_result = query_l2_memory(prompt, keyword, source_graph, embedder, cross_encoder, is_benchmark)
            
            if l2_result:
                tool_output = l2_result
                fault_line = f"L2 RE-SEARCH HIT | keyword='{keyword}' | {l2_result}"
            else:
                l3_result = query_l3_wiki(keyword, embedder)
                if l3_result:
                    tool_output = l3_result
                    fault_line = f"L3 RE-SEARCH HIT | keyword='{keyword}' | {l3_result}"
                else:
                    tool_output = f"No memory hit for keyword: {keyword}"
                    fault_line = f"RE-SEARCH MISS | keyword='{keyword}'"

            if is_benchmark and tool_output and not tool_output.startswith("No memory hit"):
                state["telemetry"].setdefault("retrieved_triples", []).append(tool_output)

            push_telemetry_item(state, "memory_faults", fault_line)
            cache.add_tool_result("search_memory", tool_output)

            conversation.append({"role": "assistant", "content": content})
            conversation.append({
                "role": "user",
                "content": (
                    f"TOOL RESULT: {tool_output}\n\n"
                    "COMMAND: Search complete. Read the tool result carefully and synthesize the final answer. "
                    "Remember to follow the MODE 1 format (prose answer followed by a CLAIMS line). "
                    "If the answer is truly not in the tool result, output 'INSUFFICIENT DATA'."
                ),
            })

            final_answer = call_policy_model(conversation, state.get("selected_model", OLLAMA_MODEL))
        else:
            final_answer = content
    else:
        final_answer = content

    if is_benchmark:
        safe_print(f"   [HADES] Answer: {final_answer}\n", flush=True)

    cache.add_history_turn("assistant", final_answer)
    return final_answer
