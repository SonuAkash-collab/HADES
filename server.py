import os
import re
import json
import tempfile
import asyncio
from typing import AsyncGenerator
from contextlib import asynccontextmanager

import networkx as nx
import ollama
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from sentence_transformers import SentenceTransformer, CrossEncoder

# Internal HADES imports
import core.pipeline as pipeline
from charon.core import L1Cache
import ui.handlers as handlers
from shared.triple import KnowledgeTriple

# --- Session State ---
# Mirroring ui/components.py's init_session_state()
SESSION_STATE = {}
APPROVED_MODELS = ["qwen3:0.6b"]

def init_session_state():
    budget = pipeline.required_system_budget()
    
    initial_model = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
    if initial_model not in APPROVED_MODELS:
        initial_model = "qwen3:0.6b"
        
    SESSION_STATE.update({
        "selected_model": initial_model,
        "messages": [],
        "source_graph": None,
        "l1_cache": L1Cache(
            budgets={
                "system": budget,
                "facts": 400,
                "history": 400,
                "tools": 300,
            }
        ),
        "telemetry": {
            "l1_status": "initialized",
            "pipeline_stage": "idle",
            "tool_calls": 0,
            "memory_faults": [],
            "cerberus_log": [],
        },
        "loaded_pdf_name": None,
        "graph_html": None,
        "graph_rendered_for": None,
        "triple_source_pages": {},
        "deep_entity_resolution": False,
    })
    
    # Add system instruction
    SESSION_STATE["l1_cache"].add_system_instruction(pipeline.SYSTEM_INSTRUCTION)

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup
    app.state.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    app.state.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    init_session_state()
    yield
    # Cleanup if needed

app = FastAPI(lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---

def get_d3_graph(source_graph):
    if not source_graph:
        return {"nodes": [], "links": []}
    
    graph: nx.DiGraph = source_graph.graph
    
    # We want D3-compatible nodes/edges
    nodes = []
    for node in graph.nodes():
        nodes.append({"id": str(node), "label": str(node)})
        
    links = []
    for u, v, data in graph.edges(data=True):
        links.append({
            "source": str(u),
            "target": str(v),
            "verb": data.get("verb", "is")
        })
        
    return {"nodes": nodes, "links": links}

async def call_policy_model_stream(messages: list[dict], model_name: str) -> AsyncGenerator[str, None]:
    """Streaming wrapper for Ollama policy model calls."""
    request_messages = [dict(m) for m in messages]
    
    if "qwen3" in model_name:
        for msg in request_messages:
            if msg["role"] == "system":
                msg["content"] += "\nCRITICAL INSTRUCTION: DO NOT output <think> tags. Do not explain your reasoning. Output only the final formatted answer immediately."
                break
    
    # Add empty assistant message to start response
    request_messages.append({"role": "assistant", "content": ""})

    # Non-blocking stream from Ollama
    stream = ollama.chat(
        model=model_name,
        messages=request_messages,
        options=pipeline.OLLAMA_OPTIONS,
        stream=True
    )
    
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content

# --- Endpoints ---

@app.post("/api/settings")
async def update_settings(payload: dict):
    model = payload.get("model")
    if model:
        if model not in APPROVED_MODELS:
            raise HTTPException(status_code=400, detail="Model not compatible with HADES prompt format")
        SESSION_STATE["selected_model"] = model
    return {"status": "ok", "selected_model": SESSION_STATE.get("selected_model")}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    SESSION_STATE["telemetry"]["pipeline_stage"] = "uploading"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Since we can't touch pipeline.py, we'll mark a general processing stage
        # or simulate the sub-stages if we really wanted to.
        SESSION_STATE["telemetry"]["pipeline_stage"] = "extracting triples"
        # ... logic inside process_pdf ...
        
        triple_count, node_count = pipeline.process_pdf(
            tmp_path, 
            SESSION_STATE, 
            app.state.embedder
        )
        SESSION_STATE["telemetry"]["pipeline_stage"] = "complete"
        SESSION_STATE["loaded_pdf_name"] = file.filename
        
        graph_data = get_d3_graph(SESSION_STATE.get("source_graph"))
        
        return {
            "triple_count": triple_count,
            "node_count": node_count,
            "graph_data": graph_data
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    prompt = payload.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")

    async def event_generator():
        SESSION_STATE["telemetry"]["pipeline_stage"] = "querying"
        # --- Stage 1: Inject clean facts ---
        pipeline.inject_clean_facts_into_l1(SESSION_STATE["l1_cache"])
        
        # --- Stage 2: Add history ---
        SESSION_STATE["l1_cache"].add_history_turn("user", prompt)
        
        # --- Stage 3 & 4: L2 search and Reranking (Charon logic) ---
        SESSION_STATE["telemetry"]["pipeline_stage"] = "compressing (Charon)"
        embedder = app.state.embedder
        cross_encoder = app.state.cross_encoder
        source_graph = SESSION_STATE.get("source_graph")
        
        active_facts = [entry.text for entry in SESSION_STATE["l1_cache"].set_facts.values()]
        l2_result = pipeline.query_l2_memory(prompt, prompt, source_graph, embedder, cross_encoder)
        l2_facts = l2_result.split("\n") if l2_result else []
        
        combined = list(set(active_facts + l2_facts))
        if combined:
            pairs = [[prompt, f] for f in combined]
            scores = cross_encoder.predict(pairs)
            scored_facts = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
            forced_facts = [f for f, s in scored_facts[:5]]
        else:
            forced_facts = []

        # --- Stage 5: Initial generation (streaming) ---
        conversation = pipeline.build_partitioned_messages(SESSION_STATE["l1_cache"], prompt, forced_facts=forced_facts)
        
        full_content = ""
        model_name = SESSION_STATE.get("selected_model", "qwen3:0.6b")
        
        async for token in call_policy_model_stream(conversation, model_name):
            full_content += token
            yield {"data": json.dumps({"token": token})}

        # --- Stage 6: Tool call detection ---
        keyword = pipeline.extract_search_keyword(full_content)
        if keyword:
            SESSION_STATE["telemetry"]["tool_calls"] += 1
            l2_result = pipeline.query_l2_memory(prompt, keyword, source_graph, embedder, cross_encoder)
            
            if l2_result:
                tool_output = l2_result
                pipeline.push_telemetry_item(SESSION_STATE, "memory_faults", f"L2 RE-SEARCH HIT | keyword='{keyword}'")
            else:
                tool_output = pipeline.query_l3_wiki(keyword, embedder)
                if tool_output:
                    pipeline.push_telemetry_item(SESSION_STATE, "memory_faults", f"L3 RE-SEARCH HIT | keyword='{keyword}'")
                else:
                    tool_output = f"No memory hit for keyword: {keyword}"
                    pipeline.push_telemetry_item(SESSION_STATE, "memory_faults", f"RE-SEARCH MISS | keyword='{keyword}'")
            
            SESSION_STATE["l1_cache"].add_tool_result("search_memory", tool_output)
            
            # Call model again with tool result
            conversation.append({"role": "assistant", "content": full_content})
            conversation.append({
                "role": "user",
                "content": f"TOOL RESULT: {tool_output}\n\nSynthesize final answer."
            })
            
            # Stream the second pass
            full_content = "" # Reset for final answer
            async for token in call_policy_model_stream(conversation, model_name):
                full_content += token
                yield {"data": json.dumps({"token": token})}

        # --- Stage 7: Cerberus Verification & L3 Writeback ---
        SESSION_STATE["telemetry"]["pipeline_stage"] = "verifying (Cerberus)"
        is_clean = pipeline.run_cerberus_writeback(full_content, SESSION_STATE)
        
        SESSION_STATE["telemetry"]["pipeline_stage"] = "writing back"
        # --- Stage 8: Post-processing & Final Cache update ---
        if not is_clean:
             yield {"data": json.dumps({"error": "Cerberus Gate Blocked Answer"})}
        
        SESSION_STATE["l1_cache"].add_history_turn("assistant", full_content)
        SESSION_STATE["telemetry"]["pipeline_stage"] = "complete"
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())

@app.get("/api/graph/state")
async def get_graph_state():
    cache: L1Cache = SESSION_STATE["l1_cache"]
    l1_nodes = list(cache.set_facts.keys())
    # For active_node, we can just pick the last one mentioned or leave as placeholder
    active_node = l1_nodes[-1] if l1_nodes else None
    
    return {
        "l1_nodes": l1_nodes,
        "active_node": active_node
    }

@app.get("/api/telemetry")
async def get_telemetry():
    return SESSION_STATE["telemetry"]

@app.post("/api/cache/flush")
async def flush_cache():
    # Mirroring handlers.handle_cache_clear() but for SESSION_STATE
    cache: L1Cache = SESSION_STATE["l1_cache"]
    cache.set_facts.clear()
    cache.set_history.clear()
    cache.set_tools.clear()
    SESSION_STATE["telemetry"]["l1_status"] = "flushed"
    return SESSION_STATE["telemetry"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
