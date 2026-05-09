import streamlit as st
import tempfile
import os
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import core.pipeline as pipeline

@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def handle_pdf_upload(uploaded_pdf):
    """Handles PDF ingestion with UI feedback."""
    if uploaded_pdf is None:
        return
    
    with st.spinner("Ingesting PDF into L2 and populating L1 facts..."):
        # Save uploaded file to a temporary location for the pipeline to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_path = tmp_file.name
        
        try:
            triple_count, node_count = pipeline.process_pdf(
                tmp_path, 
                st.session_state, 
                get_embedder()
            )
            st.session_state.loaded_pdf_name = uploaded_pdf.name
            st.session_state.graph_rendered_for = None
            st.success(f"Loaded {uploaded_pdf.name}: {triple_count} triples, {node_count} L2 graph nodes")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

def handle_query_submit(prompt):
    """Handles the chat interaction loop and Cerberus verification."""
    if not prompt:
        return

    # User message is already appended and rendered in the main app block to avoid double-rendering
    
    try:
        with st.status("NMMU thinking...", expanded=True) as status:
            status.write("Updating conversation history in L1 cache")
            status.write("Generating context and calling policy model")
            
            final_answer = pipeline.chat_loop(
                prompt, 
                st.session_state, 
                get_embedder(), 
                get_cross_encoder()
            )
            
            status.write("Running Cerberus verification and L3 write-back")
            is_clean = pipeline.run_cerberus_writeback(final_answer, st.session_state)
            
            if is_clean:
                # Clean up structured output (CLAIMS) only for clean answers
                final_answer = re.sub(r'\nCLAIMS:.*$', '', final_answer, flags=re.DOTALL).strip()
            else:
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

    st.session_state.messages.append({"role": "assistant", "content": final_answer})

def handle_cache_clear() -> None:
    """Pure state mutation for clearing the L1 cache."""
    st.session_state.l1_cache.set_facts.clear()
    st.session_state.l1_cache.set_history.clear()
    st.session_state.l1_cache.set_tools.clear()
    st.session_state.telemetry["l1_status"] = "flushed"
