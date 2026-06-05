import streamlit as st

from charon.core import L1Cache
from core.pipeline import required_system_budget, SYSTEM_INSTRUCTION
import ui.handlers as handlers

def render_custom_css():
    # Inject custom CSS for styling
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

def init_session_state():
    budget = required_system_budget()
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "qwen3:0.6b"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "source_graph" not in st.session_state:
        st.session_state.source_graph = None

    if "l1_cache" not in st.session_state:
        st.session_state.l1_cache = L1Cache(
            budgets={
                "system": budget,
                "facts": 400,
                "history": 400,
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

    # Sync cache budget if instruction set changed
    cache: L1Cache = st.session_state.l1_cache
    if cache.budgets.get("system", 0) < budget:
        st.session_state.l1_cache.add_system_instruction(SYSTEM_INSTRUCTION)

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.12em; color:#4d9fff; padding:0.5rem 0; border-bottom:1px solid #2a2a3a; margin-bottom:0.5rem;">
  ▸ HADES Telemetry
</div>
""", unsafe_allow_html=True)
        st.divider()

        # Telemetry Cards
        source_graph = st.session_state.source_graph
        active_nodes = source_graph.graph.number_of_nodes() if source_graph is not None else 0
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">Active L2 Node Density</div>
                <div class="telemetry-value">{active_nodes}</div>
            </div>
            """, unsafe_allow_html=True)

        l1_status = st.session_state.telemetry.get("l1_status", "idle").upper()
        st.markdown(f"""
            <div class="telemetry-card">
                <div class="telemetry-label">L1 Context Status</div>
                <div class="telemetry-value">{l1_status} <span class="status-tag">Live</span></div>
            </div>
            """, unsafe_allow_html=True)

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
        approved_models = ["qwen3:0.6b"]
        selected = st.selectbox(
            label="model",
            options=approved_models,
            index=approved_models.index(st.session_state.get("selected_model", "qwen3:0.6b")) if st.session_state.get("selected_model") in approved_models else 0,
            label_visibility="collapsed"
        )
        if selected != st.session_state.get("selected_model"):
            st.session_state.selected_model = selected
            st.rerun()

        st.markdown("<div class='telemetry-label'>Graph Options</div>", unsafe_allow_html=True)
        deep_res = st.toggle("Deep entity resolution", value=st.session_state.deep_entity_resolution)
        st.session_state.deep_entity_resolution = deep_res
        if deep_res:
            st.markdown("<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;color:#ffab40;padding:2px 0'>⚠ Deep mode: ingestion ~8s slower</div>", unsafe_allow_html=True)

        st.divider()
        if st.button("Flush L1 Cache"):
            handlers.handle_cache_clear()
            st.rerun()

        st.divider()
        st.markdown("<div class='telemetry-label' style='color:#b388ff'>Cerberus Gate Log</div>", unsafe_allow_html=True)
        for entry in st.session_state.telemetry.get("cerberus_log", [])[-5:]:
            colour = "#00e676" if "CLEAN" in entry else "#ff4444" if "CONTRADICTION" in entry else "#ffab40"
            short = entry[:60] + "…" if len(entry) > 60 else entry
            st.markdown(f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.62rem;color:{colour};padding:2px 0;border-left:2px solid {colour};padding-left:6px;margin:2px 0'>{short}</div>", unsafe_allow_html=True)

    with st.sidebar.expander("L1 CACHE PARTITIONS"):
        cache = st.session_state.l1_cache
        st.markdown("**System**")
        st.write(cache.set_system or ["<empty>"])
        st.markdown("**Facts**")
        st.write([entry.text for entry in cache.set_facts.values()] or ["<empty>"])
        st.markdown("**History**")
        st.write([f"{turn.role}: {turn.text}" for turn in cache.set_history] or ["<empty>"])
        st.markdown("**Tools**")
        st.write([f"{item.tool_name}: {item.text}" for item in cache.set_tools] or ["<empty>"])

def render_graph_visual(source_graph, cerberus_log):
    import networkx as nx
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    graph: nx.DiGraph = source_graph.graph
    if graph.number_of_nodes() > 150:
        try:
            pr = nx.pagerank(graph)
            top_nodes = sorted(pr, key=pr.get, reverse=True)[:150]
            subgraph = graph.subgraph(top_nodes)
        except Exception:
            subgraph = graph
    else:
        subgraph = graph

    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    try:
        pr_scores = nx.pagerank(subgraph)
    except Exception:
        pr_scores = {}
    max_pr = max(pr_scores.values()) if pr_scores else 1.0

    node_status: dict[str, str] = {}
    for entry in cerberus_log:
        status = "clean" if "clean" in entry.lower() else "dirty" if "dirty" in entry.lower() else None
        if not status: continue
        parts = entry.split("|")
        if len(parts) >= 2:
            for word in parts[1].strip().split():
                if word.strip(): node_status.setdefault(word.strip(), status)

    COLOR_CLEAN, COLOR_DIRTY, COLOR_DEFAULT = "#00c853", "#ff1744", "#448aff"
    node_colors, node_sizes, labels = [], [], {}

    for node in subgraph.nodes:
        label = str(node)
        pr = pr_scores.get(node, 0.0)
        node_sizes.append(100 + 1000 * (pr / max_pr) if max_pr else 300)
        status = node_status.get(label)
        node_colors.append(COLOR_CLEAN if status == "clean" else COLOR_DIRTY if status == "dirty" else COLOR_DEFAULT)
        labels[node] = label if len(label) <= 15 else label[:13] + "…"

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    nx.draw_networkx_edges(subgraph, pos, ax=ax, edge_color="#3d3d5c", arrows=True, arrowsize=12, alpha=0.6, node_size=node_sizes)
    nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors="#e8e8f0", linewidths=0.5)
    nx.draw_networkx_labels(subgraph, pos, labels=labels, ax=ax, font_size=8, font_color="#fafafa", font_family="sans-serif")

    plt.axis("off")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
