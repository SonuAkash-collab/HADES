import streamlit as st
import ui.components as components
import ui.handlers as handlers

# 1. Page Configuration
st.set_page_config(
    layout="wide",
    page_title="HADES",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# Main Application Entry
def main():
    # 2. UI Lifecycle & Sidebar
    components.render_custom_css()
    components.init_session_state()
    components.render_sidebar()

    st.markdown("""
<div style="display:flex; align-items:baseline; gap:1rem; border-bottom:1px solid #2a2a3a; padding-bottom:0.75rem; margin-bottom:1rem;">
  <span style="font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#e8e8f0; letter-spacing:-0.02em;">HADES</span>
  <span style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#555570; text-transform:uppercase; letter-spacing:0.1em;">
    Hierarchical Adaptive Document Encoding System · Charon Compression · Cerberus Verification · L1/L2/L3 Memory
  </span>
</div>
""", unsafe_allow_html=True)

    # 3. PDF Ingestion
    uploaded_pdf = st.file_uploader("Upload source PDF", type=["pdf"])
    if uploaded_pdf is not None and st.session_state.loaded_pdf_name != uploaded_pdf.name:
        handlers.handle_pdf_upload(uploaded_pdf)

    # 4. Main Application Tabs setup
    tab_chat, tab_map = st.tabs(["Chat", "Knowledge Map"])

    with tab_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question about the loaded document")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            handlers.handle_query_submit(prompt)
            st.rerun()

    with tab_map:
        source_graph = st.session_state.source_graph
        if source_graph is None:
            st.info("Upload a PDF to visualize the L2 Knowledge Map.")
        else:
            if st.session_state.graph_rendered_for != st.session_state.loaded_pdf_name:
                st.session_state.graph_image = components.render_graph_visual(
                    source_graph, 
                    st.session_state.telemetry.get("cerberus_log", [])
                )
                st.session_state.graph_rendered_for = st.session_state.loaded_pdf_name
            st.image(st.session_state.graph_image, use_container_width=True)

if __name__ == "__main__":
    main()