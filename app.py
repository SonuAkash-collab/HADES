from __future__ import annotations

import json
import os
import re
import tempfile

import networkx as nx
import ollama
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from charon.core import L1Cache, rank_triples_by_importance
from cerberus.core import build_source_graph, verify_claim
from shared.extractor import extract_claim_triples, extract_source_triples
from shared.l3_memory import fetch_clean_facts, save_fact
from shared.triple import KnowledgeTriple

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity


_EMBEDDER = None
_CROSS_ENCODER = None

@st.cache_resource
def _get_embedder_cached():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def _get_cross_encoder_cached():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_embedder():
    global _EMBEDDER
    if st.runtime.exists():
        return _get_embedder_cached()
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBEDDER

def get_cross_encoder():
    global _CROSS_ENCODER
    if st.runtime.exists():
        return _get_cross_encoder_cached()
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _CROSS_ENCODER


if st.runtime.exists():
    st.set_page_config(
        layout="wide",
        page_title="HADES",
        page_icon="🧠",
        initial_sidebar_state="expanded",
    )


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
    "num_predict": 512,  # Increased to prevent truncating complex tool payloads
    "repeat_penalty": 1.05,
}
FORCE_SEARCH_PROMPT = "CACHE MISS. You must use the search_memory tool now."


def _required_system_budget() -> int:
    # Keep pinned system memory large enough for the full instruction with headroom.
    return max(128, len(SYSTEM_INSTRUCTION.split()) * 3)


def _init_session_state() -> None:
    desired_system_budget = _required_system_budget()

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "qwen3:0.6b"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "source_graph" not in st.session_state:
        st.session_state.source_graph = None

    if "l1_cache" not in st.session_state:
        st.session_state.l1_cache = L1Cache(
            budgets={
                "system": desired_system_budget,
                "facts": 400,     # Expanded for dense L2/L3 context (128k window)
                "history": 400,   # Expanded for longer technical conversations
                "tools": 300,
            }
        )

    if "telemetry" not in st.session_state:
        st.session_state.telemetry = {
            "l1_status": "initialized",
            "tool_calls": 0,
            "memory_faults": [],
            "cerberus_log": [],
        }

    if "loaded_pdf_name" not in st.session_state:
        st.session_state.loaded_pdf_name = None

    if "graph_html" not in st.session_state:
        st.session_state.graph_html = None

    if "graph_rendered_for" not in st.session_state:
        st.session_state.graph_rendered_for = None

    if "triple_source_pages" not in st.session_state:
        st.session_state.triple_source_pages = {}

    if "deep_entity_resolution" not in st.session_state:
        st.session_state.deep_entity_resolution = False

    cache: L1Cache = st.session_state.l1_cache
    if cache.budgets.get("system", 0) < desired_system_budget:
        upgraded_cache = L1Cache(
            budgets={
                "system": desired_system_budget,
                "facts": cache.budgets.get("facts", 400),
                "history": cache.budgets.get("history", 400),
                "tools": cache.budgets.get("tools", 300),
            }
        )

        for entry in cache.set_facts.values():
            upgraded_cache.add_fact(entry.triple, pagerank_score=entry.pagerank_score)
        for turn in cache.set_history:
            upgraded_cache.add_history_turn(turn.role, turn.text)
        for tool_result in cache.set_tools:
            upgraded_cache.add_tool_result(tool_result.tool_name, tool_result.text)

        st.session_state.l1_cache = upgraded_cache
        cache = upgraded_cache

    if not cache.set_system:
        try:
            cache.add_system_instruction(SYSTEM_INSTRUCTION)
        except ValueError:
            fallback_cache = L1Cache(
                budgets={
                    "system": desired_system_budget * 2,
                    "facts": cache.budgets.get("facts", 400),
                    "history": cache.budgets.get("history", 400),
                    "tools": cache.budgets.get("tools", 300),
                }
            )

            for entry in cache.set_facts.values():
                fallback_cache.add_fact(entry.triple, pagerank_score=entry.pagerank_score)
            for turn in cache.set_history:
                fallback_cache.add_history_turn(turn.role, turn.text)
            for tool_result in cache.set_tools:
                fallback_cache.add_tool_result(tool_result.tool_name, tool_result.text)

            fallback_cache.add_system_instruction(SYSTEM_INSTRUCTION)
            st.session_state.l1_cache = fallback_cache


def _push_telemetry_item(key: str, value: str, max_items: int = 12) -> None:
    items = st.session_state.telemetry.setdefault(key, [])
    items.append(value)
    if len(items) > max_items:
        del items[:-max_items]


def _triple_key(triple: KnowledgeTriple) -> tuple[str, str, str]:
    return (
        str(triple.subject).strip().lower(),
        str(triple.verb).strip().lower(),
        str(triple.object).strip().lower(),
    )


def _resolve_source_page(
    triple: KnowledgeTriple,
    source_page_lookup: dict[tuple[str, str, str], int],
) -> int:
    key = _triple_key(triple)
    if key in source_page_lookup:
        return source_page_lookup[key]

    subj, verb, obj = key
    for (src_subj, src_verb, src_obj), page in source_page_lookup.items():
        if src_subj == subj and src_obj == obj:
            return page
        if src_subj == subj and src_verb == verb:
            return page

    return 0


def _inject_clean_facts_into_l1(cache: L1Cache, limit: int = 64) -> int:
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

        # L3 facts are valuable context, but should be lower-priority than active PDF facts.
        cache.add_fact(triple, pagerank_score=-1.0)
        injected += 1

    return injected


def _is_single_word_reply(text: str) -> bool:
    cleaned = text.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9]+[.!?]?", cleaned))


def _extract_search_keyword(content: str) -> str | None:
    # Use regex to find the JSON block in case the model added conversational text
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
        
        # Simplify long keywords to their core entity/value
        # Strip common question prefixes that dilute the embedding
        import re as _re_kw
        simplifications = [
            (r'^(the|a|an)\s+', ''),
            (r'\s+(scientific name|common name)\s*', ' '),
            (r'^(origin|source|location)\s+of\s+', ''),
            (r'^(author|writer|creator)\s+of\s+', ''),
            (r'\s+mentioning\s+.*$', ''),
        ]
        simplified = keyword.lower().strip()
        for pattern, replacement in simplifications:
            simplified = _re_kw.sub(pattern, replacement, simplified, flags=_re_kw.IGNORECASE).strip()
        
        # Use simplified version only if it's meaningfully shorter
        # and not empty
        if simplified and len(simplified) < len(keyword) * 0.75:
            return simplified
        return keyword
    except json.JSONDecodeError:
        return None


def _build_partitioned_messages(cache: L1Cache, prompt: str, forced_facts: list[str] = None) -> list[dict[str, str]]:
    if forced_facts is not None:
        facts = forced_facts
    else:
        facts = [entry.text for entry in cache.set_facts.values()]
    
    # --- COLLISION DETECTION (CONTRASTIVE RANKING) ---
    # Identify facts with differing numerical values for the same subject/predicate
    import re
    numbers_map = {}
    collisions = []
    
    for f in facts:
        # Extract numbers (integers or decimals)
        nums = re.findall(r'\b\d+(?:[\.,]\d+)*\b', f)
        if nums:
            # Create a 'semantic key' by removing numbers from the fact
            key = re.sub(r'\b\d+(?:[\.,]\d+)*\b', 'NUM', f).strip().lower()
            if key not in numbers_map:
                numbers_map[key] = []
            numbers_map[key].append(f)
            
    for key, related_facts in numbers_map.items():
        if len(related_facts) > 1:
            # Check if numbers actually differ
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
    
    # Add non-colliding facts
    collided_set = {f for group in collisions for f in group}
    remaining_facts = [f for f in facts if f not in collided_set]
    facts_block += "\n".join(f"- {fact}" for fact in remaining_facts) if remaining_facts else ""
    if not facts_block:
        facts_block = "<empty>"
    # -------------------------------------------------

    user_turns = [turn.text for turn in cache.set_history if turn.role == "user"]
    last_two_user_turns = user_turns[-2:]
    user_turns_block = (
        "\n".join(f"- {turn}" for turn in last_two_user_turns)
        if last_two_user_turns
        else "<empty>"
    )

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


def _call_policy_model(messages: list[dict[str, str]]) -> str:
    model_name = st.session_state.get("selected_model", OLLAMA_MODEL)
    request_messages = [dict(message) for message in messages]
    
    # --- NEW: Qwen3 Thinking Suppression ---
    if "qwen3" in model_name:
        for msg in request_messages:
            if msg["role"] == "system":
                msg["content"] += "\nCRITICAL INSTRUCTION: DO NOT output <think> tags. Do not explain your reasoning. Output only the final formatted answer immediately."
                break
    # ---------------------------------------
    
    request_messages.append({"role": "assistant", "content": ""})

    response = ollama.chat(
        model=model_name,
        messages=request_messages,
        options=OLLAMA_OPTIONS,
    )
    content = response.get("message", {}).get("content", "").strip()
    
    # Clean up <think> tags just in case the model disobeys the system prompt
    import re
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    # Robust sanitation: remove any mode-locking prefixes
    content = re.sub(r"^(MODE\s*\d.*?CACHE\s*HIT\s*:\s*|MODE\s*\d.*?:\s*|CACHE\s*(HIT|MISS).*?:\s*)", "", content, flags=re.IGNORECASE).strip()
    return content


def query_l2_memory(query: str, keyword: str, source_graph) -> str:
    """
    Search L2 memory using triple-content vector index.
    
    Architecture: Two-stage retrieval.
      Stage 1 (Bi-Encoder): Cosine similarity between query and pre-embedded
               triple texts (e.g. "Apple parent taxon Rosaceae").
               This replaces the old node-label search which compared queries
               against short strings like "Rosaceae" that had poor similarity.
      Stage 2 (Cross-Encoder): Re-rank top candidates for precise arbitration.
    """
    if source_graph is None or not keyword:
        return ""

    triple_index = getattr(source_graph, 'triple_index', None)
    if not triple_index:
        return ""

    embedder = get_embedder()
    query_vector = embedder.encode(query)

    # Stage 1: Bi-Encoder — search triple-content vectors
    scored_triples = []
    for entry in triple_index:
        if entry["vector"] is not None:
            sim = cosine_similarity([query_vector], [entry["vector"]])[0][0]
            scored_triples.append((entry["text"], sim))

    # Sort by similarity, take top 15 candidates for cross-encoder
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_triples[:15]

    if not top_candidates or top_candidates[0][1] < 0.25:
        return ""

    candidate_facts = [text for text, _ in top_candidates]

    # Stage 2: Cross-Encoder re-ranking
    cross_encoder = get_cross_encoder()
    pairs = [[query, fact] for fact in candidate_facts]
    scores = cross_encoder.predict(pairs)

    scored_candidates = sorted(
        zip(candidate_facts, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # Return top 3 facts as context
    top_facts = [fact for fact, score in scored_candidates[:3]]
    return "\n".join(top_facts)


def query_l3_wiki(keyword: str) -> str:
    from shared.l3_memory import fetch_clean_facts_by_similarity
    embedder = get_embedder()

    results = fetch_clean_facts_by_similarity(
        keyword=keyword,
        embedder=embedder,
        threshold=0.55,
        limit=3
    )

    if not results:
        return ""

    lines = []
    for fact in results:
        triple_text = (
            f"{fact['subject']} {fact['verb']} {fact['object']}".strip()
        )
        source_page = int(fact.get("source_page") or 0)
        citation = f" (source_page={source_page})" if source_page > 0 else ""
        lines.append(f"{triple_text}{citation}")

    return " | ".join(lines)


def process_pdf(file) -> tuple[int, int]:
    import pymupdf4llm
    import re as _re

    # Extract layout-aware markdown from entire PDF at once
    # pymupdf4llm handles headers, footnotes, columns automatically
    # without any font size assumptions
    try:
        md_text = pymupdf4llm.to_markdown(
            file,
            page_chunks=True,  # returns list of dicts, one per page
        )
    except Exception:
        # Fallback: open from bytes if file object doesn't work directly
        import tempfile, os
        file.seek(0)
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False
        ) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        md_text = pymupdf4llm.to_markdown(
            tmp_path, page_chunks=True
        )
        os.unlink(tmp_path)

    triples: list[KnowledgeTriple] = []
    source_page_lookup: dict[tuple[str, str, str], int] = {}
    all_sentences: list[str] = []

    STOP_MARKERS = (
        "## references", "## further reading", "## see also",
        "## external links", "## bibliography", "## other websites",
        "# references", "# further reading",
    )

    for page_number, page_chunk in enumerate(md_text, start=1):
        # page_chunk is a dict with 'text' key
        page_text = page_chunk.get("text", "")

        # Stop at reference sections
        page_lower = page_text.lower()
        if any(marker in page_lower for marker in STOP_MARKERS):
            # Truncate at the stop marker
            for marker in STOP_MARKERS:
                idx = page_lower.find(marker)
                if idx != -1:
                    page_text = page_text[:idx]
                    break

        # Clean markdown artifacts
        # Remove picture placeholders
        page_text = _re.sub(r'==> picture \[.*?\] intentionally omitted <==', '', page_text)
        raw_markdown = page_text  # keep headers for triple extraction
        
        # Remove headers (## Apple, ### Botanical information)
        clean_text = _re.sub(r'^#{1,6}\s+.*$', '', page_text, flags=_re.MULTILINE)
        # Remove bold/italic markers
        clean_text = _re.sub(r'\*\*|__|\*|_', '', clean_text)
        # Remove citation markers
        clean_text = _re.sub(r'\[\s*\d+\s*\]', '', clean_text)
        # Remove image references
        clean_text = _re.sub(r'!\[.*?\]\(.*?\)', '', clean_text)
        # Normalise whitespace
        clean_text = _re.sub(r'\s+', ' ', clean_text).strip()

        if not clean_text or len(clean_text) < 30:
            continue

        # Extract sentences for Cerberus source_sentences
        page_sentences = [
            s.strip() for s in
            _re.split(r'(?<=[.!?])\s+', clean_text)
            if len(s.strip()) > 20
        ]
        all_sentences.extend(page_sentences)

        # Extract triples from raw markdown text
        page_triples = extract_source_triples(raw_markdown)
        triples.extend(page_triples)

        for triple in page_triples:
            source_page_lookup.setdefault(
                _triple_key(triple), page_number
            )

    if not triples:
        st.session_state.source_graph = None
        st.session_state.triple_source_pages = {}
        return 0, 0
    
    embedder = get_embedder()
    source_graph = build_source_graph(triples, embedder=embedder, source_sentences=all_sentences)
    st.session_state.source_graph = source_graph
    st.session_state.triple_source_pages = source_page_lookup

    cache: L1Cache = st.session_state.l1_cache
    cache.set_facts.clear()
    cache.set_tools.clear()

    ranked = rank_triples_by_importance(triples)
    for triple, score in ranked:
        cache.add_fact(triple, pagerank_score=score)

    injected_from_l3 = _inject_clean_facts_into_l1(cache)

    st.session_state.telemetry["l1_status"] = "pdf_loaded"

    # Optional deep entity resolution pass
    if st.session_state.get("deep_entity_resolution", False):
        from charon.core.graph import normalise_entities_with_llm
        with st.spinner("Deep entity resolution running..."):
            source_graph.graph, extra_merges = normalise_entities_with_llm(
                source_graph.graph,
                ollama_model=st.session_state.get("selected_model", OLLAMA_MODEL),
            )
        if extra_merges > 0:
            _push_telemetry_item(
                "memory_faults",
                f"Deep resolution: {extra_merges} additional aliases resolved"
            )

    nodes_after = source_graph.graph.number_of_nodes()
    _push_telemetry_item(
        "memory_faults",
        (
            f"PDF ingested: {len(triples)} raw triples → "
            f"{nodes_after} merged graph nodes "
            f"(L3 injected: {injected_from_l3})"
        ),
    )
    return len(triples), source_graph.graph.number_of_nodes()


def _render_sidebar() -> None:
    with st.sidebar:
        st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; 
     font-size:0.65rem; text-transform:uppercase; 
     letter-spacing:0.12em; color:#4d9fff; 
     padding:0.5rem 0; border-bottom:1px solid #2a2a3a;
     margin-bottom:0.5rem;">
  ▸ HADES Telemetry
</div>
""", unsafe_allow_html=True)
        st.divider()

        # Node Density Card
        source_graph = st.session_state.source_graph
        active_nodes = source_graph.graph.number_of_nodes() if source_graph is not None else 0
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">Active L2 Node Density</div>
                <div class="telemetry-value">{active_nodes}</div>
            </div>
            """, unsafe_allow_html=True)

        # L1 Status Card
        l1_status = st.session_state.telemetry.get("l1_status", "idle").upper()
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">L1 Context Status</div>
                <div class="telemetry-value">{l1_status} <span class="status-tag">Live</span></div>
            </div>
            """, unsafe_allow_html=True)

        # Tool Call Counters
        tool_calls = st.session_state.telemetry.get("tool_calls", 0)
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">Total Tool Faults (L2/L3)</div>
                <div class="telemetry-value">{tool_calls}</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='telemetry-label'>Memory Fault Logs</div>", unsafe_allow_html=True)
        for fault in st.session_state.telemetry.get("memory_faults", [])[-5:]:
            st.caption(f"> {fault}")

        st.divider()
        st.markdown(
            "<div class='telemetry-label'>Inference Model</div>",
            unsafe_allow_html=True,
        )
        selected = st.selectbox(
            label="model",
            options=["qwen3:0.6b", "phi3.5", "llama3.2:3b"],
            index=["qwen3:0.6b", "phi3.5", "llama3.2:3b"].index(
                st.session_state.get("selected_model", "qwen3:0.6b")
            ),
            label_visibility="collapsed",
            help=(
                "qwen3:0.6b — maximum accessibility, fix instruction failures\n"
                "phi3.5 — better reasoning, slower\n"
                "llama3.2:3b — balanced, requires more RAM"
            ),
        )
        if selected != st.session_state.get("selected_model"):
            st.session_state.selected_model = selected
            st.rerun()

        st.markdown(
            "<div class='telemetry-label'>Graph Options</div>",
            unsafe_allow_html=True,
        )
        deep_res = st.toggle(
            "Deep entity resolution",
            value=st.session_state.deep_entity_resolution,
            help=(
                "Runs a local LLM pass to resolve pronouns and "
                "implicit references (e.g. 'it' → 'apple tree'). "
                "Adds 5-8 seconds to PDF ingestion. "
                "Disable for fast demos."
            ),
        )
        st.session_state.deep_entity_resolution = deep_res
        if deep_res:
            st.markdown(
                "<div style='font-family:IBM Plex Mono,monospace;"
                "font-size:0.6rem;color:#ffab40;padding:2px 0'>"
                "⚠ Deep mode: ingestion ~8s slower</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("Flush L1 Cache"):
            st.session_state.l1_cache.set_facts.clear()
            st.session_state.l1_cache.set_history.clear()
            st.session_state.l1_cache.set_tools.clear()
            st.session_state.telemetry["l1_status"] = "flushed"
            st.rerun()

        st.divider()
        st.markdown(
            "<div class='telemetry-label' style='color:#b388ff'>Cerberus Gate Log</div>",
            unsafe_allow_html=True
        )
        for entry in st.session_state.telemetry.get("cerberus_log", [])[-5:]:
            if "CLEAN" in entry:
                colour = "#00e676"
            elif "CONTRADICTION" in entry:
                colour = "#ff4444"
            else:
                colour = "#ffab40"
            short = entry[:60] + "…" if len(entry) > 60 else entry
            st.markdown(
                f"<div style='font-family:IBM Plex Mono,monospace;"
                f"font-size:0.62rem;color:{colour};padding:2px 0;"
                f"border-left:2px solid {colour};padding-left:6px;"
                f"margin:2px 0'>{short}</div>",
                unsafe_allow_html=True
            )

    cache: L1Cache = st.session_state.l1_cache
    with st.sidebar.expander("L1 CACHE PARTITIONS", expanded=False):
        st.markdown("**System**")
        st.write(cache.set_system or ["<empty>"])

        st.markdown("**Facts**")
        st.write([entry.text for entry in cache.set_facts.values()] or ["<empty>"])

        st.markdown("**History**")
        st.write([f"{turn.role}: {turn.text}" for turn in cache.set_history] or ["<empty>"])

        st.markdown("**Tools**")
        st.write([f"{item.tool_name}: {item.text}" for item in cache.set_tools] or ["<empty>"])




def _run_cerberus_writeback(final_answer: str) -> bool:
    """Verify answer triples against the source graph.

    Returns ``True`` if no triples actively **contradict** the source.
    Returns ``False`` if any triple is labelled *contradiction* by DeBERTa,
    indicating an active hallucination.
    """
    source_graph = st.session_state.source_graph
    if source_graph is None:
        _push_telemetry_item("cerberus_log", "No source graph loaded; Cerberus verification skipped.")
        return True

    import re as _re
    import json
    answer_triples = []
    claims_match = _re.search(r'CLAIMS:\s*(\[.*?\])', final_answer, _re.DOTALL)
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
    # Fallback to spaCy if model didn't emit CLAIMS
    if not answer_triples:
        from shared.extractor import _extract_svo_triples
        answer_triples = _extract_svo_triples(final_answer)
    if not answer_triples:
        _push_telemetry_item("cerberus_log", "No triples extracted from assistant answer.")
        return True

    source_page_lookup: dict[tuple[str, str, str], int] = st.session_state.get("triple_source_pages", {})
    has_contradiction = False

    for triple in answer_triples:
        verdict = verify_claim(triple, source_graph, source_sentences=source_graph.source_sentences)
        if verdict.is_verified:
            source_page = _resolve_source_page(triple, source_page_lookup)
            inserted = save_fact(
                triple,
                source_page=source_page,
                cerberus_status="CLEAN",
            )
            citation = f"p.{source_page}" if source_page > 0 else "p.?"
            write_result = "stored" if inserted else "duplicate_ignored"
            _push_telemetry_item(
                "cerberus_log",
                f"✅ CLEAN | {triple.as_text()} | {verdict.reason} | {citation} | {write_result}",
            )
        elif verdict.label == "contradiction":
            has_contradiction = True
            _push_telemetry_item(
                "cerberus_log",
                f"❌ CONTRADICTION | {triple.as_text()} | {verdict.reason}",
            )
        else:
            # Neutral — not entailed but not contradicted either (paraphrase)
            _push_telemetry_item(
                "cerberus_log",
                f"⚠️ NEUTRAL | {triple.as_text()} | {verdict.reason}",
            )

    _inject_clean_facts_into_l1(st.session_state.l1_cache)
    return not has_contradiction


def _chat_loop(prompt: str) -> str:
    cache: L1Cache = st.session_state.l1_cache
    source_graph = st.session_state.source_graph

    _inject_clean_facts_into_l1(cache)
    cache.add_history_turn("user", prompt)

    # --- L1/L2 JOINT RETRIEVAL ---
    active_facts = [entry.text for entry in cache.set_facts.values()]
    l2_result = query_l2_memory(prompt, prompt, source_graph)
    l2_facts = l2_result.split("\n") if l2_result else []

    if not active_facts and not l2_facts:
        forced_facts = []
    elif not active_facts:
        forced_facts = l2_facts
    else:
        combined = list(set(active_facts + l2_facts))
        ce = get_cross_encoder()
        pairs = [[prompt, f] for f in combined]
        scores = ce.predict(pairs)
        scored_facts = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
        # These are the top facts actually handed to the LLM
        forced_facts = [f for f, s in scored_facts[:5]]

    # --- 1. LOG INITIAL RETRIEVAL HITS ---
    if st.session_state.telemetry.get("l1_status") == "benchmark":
        st.session_state.telemetry.setdefault("retrieved_triples", []).extend(forced_facts)
    # -------------------------------------

    conversation = _build_partitioned_messages(cache, prompt, forced_facts=forced_facts)
    content = _call_policy_model(conversation)
    
    # Allow the model to 're-search' if it detects a context gap
    keyword = _extract_search_keyword(content)
    if keyword:
        # Check if we already searched for this exact keyword in this turn to avoid loops
        already_searched = any(
            isinstance(item, dict) and item.get("keyword") == keyword 
            for item in st.session_state.telemetry.get("memory_faults", [])
        )
        
        if not already_searched:
            st.session_state.telemetry["tool_calls"] = st.session_state.telemetry.get("tool_calls", 0) + 1
            l2_result = query_l2_memory(prompt, keyword, source_graph)
            
            if l2_result:
                tool_output = l2_result
                fault_line = f"L2 RE-SEARCH HIT | keyword='{keyword}' | {l2_result}"
            else:
                l3_result = query_l3_wiki(keyword)
                if l3_result:
                    tool_output = l3_result
                    fault_line = f"L3 RE-SEARCH HIT | keyword='{keyword}' | {l3_result}"
                else:
                    tool_output = f"No memory hit for keyword: {keyword}"
                    fault_line = f"RE-SEARCH MISS | keyword='{keyword}'"

            # --- 2. LOG RE-SEARCH TOOL HITS ---
            if st.session_state.telemetry.get("l1_status") == "benchmark" and tool_output and not tool_output.startswith("No memory hit"):
                st.session_state.telemetry.setdefault("retrieved_triples", []).append(tool_output)
            # ----------------------------------

            _push_telemetry_item("memory_faults", fault_line)
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

            final_answer = _call_policy_model(conversation)
        else:
            final_answer = content
    else:
        final_answer = content

    cache.add_history_turn("assistant", final_answer)
    return final_answer


def render_graph_visual(source_graph) -> bytes:
    """Build a static Matplotlib graph from the L2 SourceGraph.

    Nodes are sized by PageRank and colored by Cerberus verification
    status: green = Clean, red = Dirty, blue = unverified.
    Returns the generated PNG image as bytes.
    """
    import networkx as nx
    import io
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
    import matplotlib.pyplot as plt

    graph: nx.DiGraph = source_graph.graph

    # If graph is too large, sample it down to prevent hairball rendering
    if graph.number_of_nodes() > 150:
        try:
            pr = nx.pagerank(graph)
            top_nodes = sorted(pr, key=pr.get, reverse=True)[:150]
            subgraph = graph.subgraph(top_nodes)
        except Exception:
            subgraph = graph
    else:
        subgraph = graph

    # Calculate layout using NetworkX
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

    # ── PageRank scores for node sizing ──
    try:
        pr_scores = nx.pagerank(subgraph)
    except Exception:
        pr_scores = {}

    max_pr = max(pr_scores.values()) if pr_scores else 1.0

    # ── Cerberus verification status lookup ──
    cerberus_log = st.session_state.telemetry.get("cerberus_log", [])
    node_status: dict[str, str] = {}  # node label -> "clean" | "dirty"
    for entry in cerberus_log:
        entry_lower = entry.lower()
        if "clean" in entry_lower:
            status = "clean"
        elif "dirty" in entry_lower:
            status = "dirty"
        else:
            continue
        parts = entry.split("|")
        if len(parts) >= 2:
            triple_text = parts[1].strip()
            for word in triple_text.split():
                cleaned = word.strip()
                if cleaned:
                    node_status.setdefault(cleaned, status)

    # ── Color palette ──
    COLOR_CLEAN = "#00c853"   # green
    COLOR_DIRTY = "#ff1744"   # red
    COLOR_DEFAULT = "#448aff" # high-contrast blue

    node_colors = []
    node_sizes = []
    labels = {}

    for node in subgraph.nodes:
        label = str(node)
        pr = pr_scores.get(node, 0.0)
        
        # Scale sizes for matplotlib
        size = 100 + 1000 * (pr / max_pr) if max_pr else 300
        node_sizes.append(size)

        status = node_status.get(label)
        if status == "clean":
            node_colors.append(COLOR_CLEAN)
        elif status == "dirty":
            node_colors.append(COLOR_DIRTY)
        else:
            node_colors.append(COLOR_DEFAULT)

        labels[node] = label if len(label) <= 15 else label[:13] + "…"

    # ── Plotting ──
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    nx.draw_networkx_edges(
        subgraph, pos, ax=ax,
        edge_color="#3d3d5c",
        arrows=True,
        arrowsize=12,
        alpha=0.6,
        node_size=node_sizes,
    )

    nx.draw_networkx_nodes(
        subgraph, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#e8e8f0",
        linewidths=0.5
    )

    nx.draw_networkx_labels(
        subgraph, pos, labels=labels, ax=ax,
        font_size=8,
        font_color="#fafafa",
        font_family="sans-serif"
    )

    plt.axis("off")
    plt.tight_layout()

    # Save to buffer and return bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def main() -> None:
    st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Root theme ── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-hover: #1e1e2a;
    --border: #2a2a3a;
    --border-accent: #3d3d5c;
    --text-primary: #e8e8f0;
    --text-secondary: #8888aa;
    --text-muted: #555570;
    --accent-blue: #4d9fff;
    --accent-green: #00e676;
    --accent-red: #ff4444;
    --accent-amber: #ffab40;
    --accent-purple: #b388ff;
    --font-mono: 'IBM Plex Mono', monospace;
    --font-sans: 'IBM Plex Sans', sans-serif;
}

/* ── Global app background ── */
.stApp {
    background-color: var(--bg-primary);
    font-family: var(--font-sans);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stSidebar"] .stMetric label {
    color: var(--text-muted) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: var(--accent-blue) !important;
    font-size: 1.4rem !important;
    font-weight: 600;
}

/* ── Main content area ── */
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1400px;
}

/* ── Title ── */
h1 {
    font-family: var(--font-mono) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.75rem;
    margin-bottom: 0.25rem !important;
}

/* ── Caption / subtitle ── */
[data-testid="stCaptionContainer"] p {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border-accent) !important;
    border-radius: 6px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] * {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    gap: 0;
    padding: 0;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted) !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    margin-bottom: 0.5rem !important;
    padding: 0.75rem 1rem !important;
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid var(--accent-blue) !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    border-left: 3px solid var(--accent-green) !important;
}
[data-testid="stChatMessage"] p {
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 6px !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-sans) !important;
    font-size: 0.88rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Status boxes (thinking indicator) ── */
[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-secondary) !important;
}

/* ── Success / error / info alerts ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    padding: 0.5rem 0.75rem !important;
}

/* ── Expander (L1 cache view in sidebar) ── */
[data-testid="stExpander"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: var(--border-accent);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: var(--accent-blue) !important;
}

/* ── Sidebar section headers ── */
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent-purple) !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.3rem;
    margin-top: 1rem !important;
}

/* ── Telemetry cards ── */
.telemetry-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.4rem 0.6rem;
    margin-bottom: 0.4rem;
}
.telemetry-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}
.telemetry-value {
    font-family: var(--font-mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent-blue);
    margin-top: 0.1rem;
}
.status-tag {
    font-size: 0.55rem;
    background: var(--accent-green);
    color: #000;
    padding: 1px 5px;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 4px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

    _init_session_state()
    _render_sidebar()

    st.markdown("""
<div style="display:flex; align-items:baseline; gap:1rem; 
     border-bottom:1px solid #2a2a3a; padding-bottom:0.75rem; 
     margin-bottom:1rem;">
  <span style="font-family:'IBM Plex Mono',monospace; 
               font-size:1.4rem; font-weight:600; 
               color:#e8e8f0; letter-spacing:-0.02em;">
    HADES
  </span>
  <span style="font-family:'IBM Plex Mono',monospace; 
               font-size:0.65rem; color:#555570; 
               text-transform:uppercase; letter-spacing:0.1em;">
    Hierarchical Adaptive Document Encoding System · Charon Compression · Cerberus Verification · L1/L2/L3 Memory
  </span>
</div>
""", unsafe_allow_html=True)

    uploaded_pdf = st.file_uploader("Upload source PDF", type=["pdf"])
    if uploaded_pdf is not None and st.session_state.loaded_pdf_name != uploaded_pdf.name:
        with st.spinner("Ingesting PDF into L2 and populating L1 facts..."):
            triple_count, node_count = process_pdf(uploaded_pdf)
            st.session_state.loaded_pdf_name = uploaded_pdf.name
            # Invalidate cached graph rendering so the map tab re-renders
            st.session_state.graph_rendered_for = None
        st.success(f"Loaded {uploaded_pdf.name}: {triple_count} triples, {node_count} L2 graph nodes")

    # ── Tabbed layout: Chat + Knowledge Map ──
    tab_chat, tab_map = st.tabs(["Chat", "Knowledge Map"])

    # ── Chat tab ──
    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question about the loaded document")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                try:
                    with st.status("NMMU thinking...", expanded=True) as status:
                        status.write("Updating conversation history in L1 cache")
                        status.write("Generating context and calling policy model")
                        final_answer = _chat_loop(prompt)
                        status.write("Running Cerberus verification and L3 write-back")
                        is_clean = _run_cerberus_writeback(final_answer)
                        if not is_clean:
                            final_answer = (
                                "🚨 CERBERUS GATE BLOCK: My policy engine attempted to answer this, "
                                "but the local DeBERTa-v3 verification failed. The source document "
                                "does not support this claim."
                            )
                        status.update(label="NMMU complete", state="complete")
                except Exception as exc:
                    final_answer = f"Error: {exc}"
                    st.error(final_answer)
                    st.session_state.telemetry["l1_status"] = "error"

                import re as _re
                final_answer = _re.sub(r'\nCLAIMS:.*$', '', final_answer, flags=_re.DOTALL).strip()
                answer_placeholder.markdown(final_answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})

    # ── Knowledge Map tab ──
    with tab_map:
        source_graph = st.session_state.source_graph
        if source_graph is None:
            st.info("Upload a PDF to visualize the L2 Knowledge Map.")
        else:
            # Only re-render if the source graph has changed
            current_pdf = st.session_state.loaded_pdf_name
            if st.session_state.graph_rendered_for != current_pdf:
                # Store the image bytes instead of HTML
                st.session_state.graph_image = render_graph_visual(source_graph)
                st.session_state.graph_rendered_for = current_pdf

            # Render the static image instantly
            st.image(st.session_state.graph_image, use_container_width=True)


if __name__ == "__main__":
    main()