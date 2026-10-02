import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

import config
from rag.document_loader import DocumentLoader, DocumentChunk
from rag.text_splitter import RecursiveCharacterSplitter
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="DocuGemini | AI Document Intelligence & RAG",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Modern Theme Accents */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4285F4, #9B51E0, #34A853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #9AA0A6;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4285F4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9AA0A6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .citation-card {
        background: rgba(66, 133, 244, 0.08);
        border-left: 4px solid #4285F4;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .badge-score {
        background-color: #34A853;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-doc {
        background-color: #9B51E0;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-page {
        background-color: #EA4335;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = ""
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = ""

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Gemini & RAG Setup")
    
    # API Key Input
    saved_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=saved_key,
        type="password",
        help="Get your free API key at https://aistudio.google.com/"
    )
    
    # Model Selector
    selected_model = st.selectbox(
        "Gemini Generation Model",
        options=config.AVAILABLE_CHAT_MODELS,
        index=0
    )
    
    with st.expander("🛠️ Advanced RAG Parameters", expanded=False):
        temperature = st.slider("Temperature (Creativity vs Strictness)", 0.0, 1.0, config.DEFAULT_TEMPERATURE, 0.05)
        top_k = st.slider("Top-K Retrieved Chunks", 1, 10, config.DEFAULT_TOP_K, 1)
        chunk_size = st.slider("Chunk Size (Characters)", 200, 2000, config.DEFAULT_CHUNK_SIZE, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, config.DEFAULT_CHUNK_OVERLAP, 25)

    st.markdown("---")
    st.markdown("### 📂 Document Ingestion")
    
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supports PDF (with page tracking), DOCX, TXT, Markdown, CSV"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        ingest_btn = st.button("🚀 Ingest Files", use_container_width=True, type="primary")
    with col_btn2:
        sample_btn = st.button("📁 Load Samples", use_container_width=True)

    # Initialize Managers
    embedding_manager = EmbeddingManager(api_key=api_key)
    vector_store = st.session_state.vector_store
    rag_engine = RAGEngine(api_key=api_key, model_name=selected_model, temperature=temperature)

    # Ingestion Logic
    if ingest_btn and uploaded_files:
        with st.spinner("Parsing and chunking documents..."):
            splitter = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            total_added = 0
            for uploaded_file in uploaded_files:
                try:
                    raw_chunks = DocumentLoader.load_file(uploaded_file)
                    split_chunks = splitter.split_documents(raw_chunks)
                    added = vector_store.add_documents(split_chunks, embedding_manager)
                    total_added += added
                except Exception as e:
                    st.error(f"Error reading {uploaded_file.name}: {e}")

            st.success(f"✅ Successfully indexed {total_added} chunks across {len(uploaded_files)} files!")
            st.rerun()

    if sample_btn:
        with st.spinner("Loading sample reports..."):
            splitter = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            sample_paths = [
                "samples/sample_report.txt",
                "samples/ai_overview.txt"
            ]
            total_added = 0
            for path in sample_paths:
                if os.path.exists(path):
                    raw_chunks = DocumentLoader.load_file(path)
                    split_chunks = splitter.split_documents(raw_chunks)
                    added = vector_store.add_documents(split_chunks, embedding_manager)
                    total_added += added
            st.success(f"✅ Ingested {total_added} sample chunks into Vector Database!")
            st.rerun()

    # Document Library Manager
    st.markdown("---")
    st.markdown("### 📚 Indexed Library")
    stats = vector_store.get_stats()
    indexed_sources = stats.get("documents", [])
    
    if indexed_sources:
        for src in indexed_sources:
            col_doc, col_del = st.columns([4, 1])
            with col_doc:
                st.caption(f"📄 **{src}**")
            with col_del:
                if st.button("🗑️", key=f"del_{src}", help=f"Delete {src}"):
                    vector_store.delete_source(src)
                    st.toast(f"Removed {src}")
                    st.rerun()
        
        if st.button("⚠️ Clear Entire Database", use_container_width=True):
            vector_store.clear()
            st.session_state.chat_history = []
            st.session_state.analysis_result = ""
            st.session_state.comparison_result = ""
            st.success("Vector Store Cleared!")
            st.rerun()
    else:
        st.info("No documents currently indexed. Upload files or click 'Load Samples'.")

# Main Interface Layout
st.markdown('<div class="main-title">DocuGemini AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Advanced Multimodal Retrieval-Augmented Generation & Deep Document Intelligence</div>', unsafe_allow_html=True)

# Top Status Metrics Cards
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
with stat_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats.get('document_count', 0)}</div>
        <div class="metric-label">Indexed Documents</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats.get('total_chunks', 0)}</div>
        <div class="metric-label">Vector Chunks</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{selected_model.split('-')[1].capitalize() if '-' in selected_model else selected_model}</div>
        <div class="metric-label">Active Gemini Model</div>
    </div>
    """, unsafe_allow_html=True)

with stat_col4:
    status_text = "🟢 Ready" if api_key and stats.get('total_chunks', 0) > 0 else ("🟡 Key Required" if not api_key else "🔵 Awaiting Docs")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size: 1.4rem;">{status_text}</div>
        <div class="metric-label">System Status</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_chat, tab_analysis, tab_compare, tab_search = st.tabs([
    "💬 Interactive Q&A Chat",
    "📊 Deep Document Analysis",
    "📑 Document Comparison",
    "🔍 Vector Search Explorer"
])

# -------------------------------------------------------------
# TAB 1: INTERACTIVE CHAT (RAG Q&A)
# -------------------------------------------------------------
with tab_chat:
    st.markdown("#### 💬 Grounded Document Q&A with Strict Page Citations")
    st.caption("Ask questions about your uploaded documents. Answers are strictly grounded in retrieved vector chunks with exact citations.")

    # Quick Prompt Shortcuts
    if indexed_sources:
        st.markdown("**Quick Prompts:**")
        qp_col1, qp_col2, qp_col3, qp_col4 = st.columns(4)
        quick_query = None
        with qp_col1:
            if st.button("💡 Summarize Key Findings", use_container_width=True):
                quick_query = "What are the key findings and major takeaways across the documents?"
        with qp_col2:
            if st.button("📅 Milestones & Deadlines", use_container_width=True):
                quick_query = "List all important dates, timelines, and milestones mentioned."
        with qp_col3:
            if st.button("⚠️ Risks & Challenges", use_container_width=True):
                quick_query = "What are the primary risks, bottlenecks, or challenges identified?"
        with qp_col4:
            if st.button("📊 Financial & Key Metrics", use_container_width=True):
                quick_query = "What are the key financial numbers, investments, and quantitative metrics?"
    else:
        quick_query = None

    # Display Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander(f"📚 View Citations ({len(msg['citations'])} Source Snippets)", expanded=False):
                    for cit in msg["citations"]:
                        st.markdown(f"""
                        <div class="citation-card">
                            <span class="badge-doc">📄 {cit['source']}</span>
                            <span class="badge-page">Page {cit['page']}</span>
                            <span class="badge-score">Match: {cit['similarity']}%</span>
                            <p style="margin-top: 0.5rem; color: #E8EAED;">{cit['snippet']}</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input Handling
    user_query = st.chat_input("Ask a question about your documents...") or quick_query

    if user_query:
        if not api_key:
            st.error("⚠️ Please provide a Gemini API Key in the sidebar to generate answers.")
        elif stats.get("total_chunks", 0) == 0:
            st.warning("⚠️ No documents indexed yet. Please upload files or load sample documents in the sidebar.")
        else:
            # Append User Message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # RAG Retrieval and Answer Generation
            with st.chat_message("assistant"):
                with st.spinner("Searching vector index & generating grounded answer..."):
                    retrieved_chunks = vector_store.query(
                        query_text=user_query,
                        top_k=top_k,
                        embedding_manager=embedding_manager
                    )
                    rag_response = rag_engine.answer_query(
                        query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        chat_history=st.session_state.chat_history[:-1]
                    )

                st.markdown(rag_response["answer"])

                if rag_response.get("citations"):
                    with st.expander(f"📚 View Citations ({len(rag_response['citations'])} Source Snippets)", expanded=False):
                        for cit in rag_response["citations"]:
                            st.markdown(f"""
                            <div class="citation-card">
                                <span class="badge-doc">📄 {cit['source']}</span>
                                <span class="badge-page">Page {cit['page']}</span>
                                <span class="badge-score">Match: {cit['similarity']}%</span>
                                <p style="margin-top: 0.5rem; color: #E8EAED;">{cit['snippet']}</p>
                            </div>
                            """, unsafe_allow_html=True)

            # Save Assistant Response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": rag_response["answer"],
                "citations": rag_response.get("citations", [])
            })

    # Chat export & clear controls
    if st.session_state.chat_history:
        st.write("")
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        with c2:
            chat_export_text = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button("📥 Export JSON", data=chat_export_text, file_name="chat_history.json", mime="application/json")

# -------------------------------------------------------------
# TAB 2: DEEP DOCUMENT ANALYSIS
# -------------------------------------------------------------
with tab_analysis:
    st.markdown("#### 📊 Deep Automated Document Intelligence")
    st.caption("Generate structured executive summaries, extract domain entities, or create comprehension quizzes.")

    if not indexed_sources:
        st.info("Please index documents to use automated deep analysis.")
    else:
        doc_options = ["All Indexed Documents"] + indexed_sources
        selected_doc_analysis = st.selectbox("Select Target Document for Analysis", options=doc_options)
        
        filter_doc = None if selected_doc_analysis == "All Indexed Documents" else selected_doc_analysis

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            gen_summary = st.button("📌 Executive Summary", use_container_width=True, type="primary")
        with btn_col2:
            gen_entities = st.button("🏷️ Extract Entities & Insights", use_container_width=True)
        with btn_col3:
            gen_quiz = st.button("🧩 Comprehension Quiz", use_container_width=True)

        if gen_summary:
            with st.spinner("Generating executive summary..."):
                retrieved = vector_store.query("executive overview main themes findings metrics risks recommendations", top_k=8, embedding_manager=embedding_manager, filter_source=filter_doc)
                st.session_state.analysis_result = rag_engine.generate_summary(retrieved, selected_doc_analysis)

        if gen_entities:
            with st.spinner("Extracting structured entities & data points..."):
                retrieved = vector_store.query("organizations stakeholders companies dates timelines numbers statistics", top_k=8, embedding_manager=embedding_manager, filter_source=filter_doc)
                st.session_state.analysis_result = rag_engine.extract_entities_and_insights(retrieved)

        if gen_quiz:
            with st.spinner("Synthesizing comprehension questions..."):
                retrieved = vector_store.query("key concepts definitions findings facts conclusions", top_k=8, embedding_manager=embedding_manager, filter_source=filter_doc)
                st.session_state.analysis_result = rag_engine.generate_quiz(retrieved)

        if st.session_state.analysis_result:
            st.markdown("---")
            st.markdown("### 📋 Analysis Report")
            st.markdown(st.session_state.analysis_result)
            st.download_button(
                "📥 Download Analysis Report (Markdown)",
                data=st.session_state.analysis_result,
                file_name=f"analysis_report_{selected_doc_analysis.replace(' ', '_')}.md",
                mime="text/markdown"
            )

# -------------------------------------------------------------
# TAB 3: DOCUMENT COMPARISON
# -------------------------------------------------------------
with tab_compare:
    st.markdown("#### 📑 Multi-Document Comparative Matrix")
    st.caption("Compare two or more documents to identify overlapping themes, strategic contrasts, and differing conclusions.")

    if len(indexed_sources) < 2:
        st.warning("⚠️ Please index at least 2 documents to perform comparative analysis.")
    else:
        selected_compare_docs = st.multiselect(
            "Select Documents to Compare",
            options=indexed_sources,
            default=indexed_sources[:2]
        )

        if st.button("⚡ Run Comparative Synthesis", type="primary"):
            if len(selected_compare_docs) < 2:
                st.error("Please select at least 2 documents.")
            else:
                with st.spinner("Analyzing document intersections & contrasts..."):
                    doc_chunks_dict = {}
                    for doc_name in selected_compare_docs:
                        chunks = vector_store.query(
                            query_text="core purpose methodologies findings metrics conclusions risks",
                            top_k=4,
                            embedding_manager=embedding_manager,
                            filter_source=doc_name
                        )
                        doc_chunks_dict[doc_name] = chunks

                    st.session_state.comparison_result = rag_engine.compare_documents(doc_chunks_dict)

        if st.session_state.comparison_result:
            st.markdown("---")
            st.markdown("### 📑 Comparative Analysis Report")
            st.markdown(st.session_state.comparison_result)
            st.download_button(
                "📥 Download Comparison Report",
                data=st.session_state.comparison_result,
                file_name="comparative_report.md",
                mime="text/markdown"
            )

# -------------------------------------------------------------
# TAB 4: VECTOR SEARCH EXPLORER
# -------------------------------------------------------------
with tab_search:
    st.markdown("#### 🔍 Vector Index & Chunk Explorer")
    st.caption("Directly query the vector database to inspect chunk embeddings, similarity scores, and metadata.")

    col_q1, col_q2, col_q3 = st.columns([3, 1, 1])
    with col_q1:
        search_query = st.text_input("Semantic Search Query", placeholder="e.g. Battery storage capacity or data privacy")
    with col_q2:
        search_top_k = st.number_input("Top Chunks", min_value=1, max_value=20, value=5)
    with col_q3:
        filter_source_search = st.selectbox("Source Filter", options=["All"] + indexed_sources)

    if search_query:
        filter_src = None if filter_source_search == "All" else filter_source_search
        results = vector_store.query(
            query_text=search_query,
            top_k=search_top_k,
            embedding_manager=embedding_manager,
            filter_source=filter_src
        )

        if results:
            st.write(f"Found **{len(results)}** matching chunks:")
            for i, res in enumerate(results):
                meta = res.get("metadata", {})
                score = res.get("similarity_score", 0)
                with st.expander(f"Chunk #{i+1} | {meta.get('source', 'Unknown')} | Page {meta.get('page', 1)} | Similarity: {score}%", expanded=(i == 0)):
                    st.progress(score / 100.0)
                    st.markdown(f"**Chunk ID:** `{meta.get('chunk_id', 'N/A')}` | **Characters:** {meta.get('char_length', len(res['text']))}")
                    st.text_area("Chunk Content", value=res["text"], height=140, key=f"raw_chunk_{i}")
        else:
            st.info("No matching chunks found for your query.")
