import uuid
import streamlit as st

from main import ask_ollama, load_config
from rag import build_index, pdf_to_chunks, retrieve_chunks

# Page configuration
st.set_page_config(
    page_title="NLP Chatbot - Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Global subtle improvements */
    .stApp {
        background-color: var(--background-color);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Container Cards */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.05) 0%, rgba(99, 102, 241, 0.08) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .stat-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #4f46e5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    .stat-card-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }

    .stat-card-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: inherit;
    }

    /* Chat bubble enhancements */
    .chat-source-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-right: 6px;
        margin-top: 4px;
    }

    .sample-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.06);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.8rem;
        cursor: pointer;
        margin: 4px;
        transition: all 0.2s;
    }
    .sample-chip:hover {
        background: rgba(99, 102, 241, 0.15);
        border-color: #6366f1;
    }

    /* File upload area clean styling */
    .upload-container {
        border: 1px dashed rgba(148, 163, 184, 0.35);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load configuration
try:
    config = load_config()
except (OSError, ValueError) as error:
    st.error(f"⚠️ Could not load chatbot configuration: {error}")
    st.stop()

# Session State initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_collection_name" not in st.session_state:
    st.session_state.rag_collection_name = None
if "indexed_upload_signature" not in st.session_state:
    st.session_state.indexed_upload_signature = None
if "indexed_sources" not in st.session_state:
    st.session_state.indexed_sources = []
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


@st.cache_resource
def get_vector_client():
    import chromadb

    return chromadb.EphemeralClient()


# SIDEBAR: Configuration & Knowledge Base Management
with st.sidebar:
    st.markdown("### ⚡ System Status")
    
    col_dot, col_status = st.columns([1, 6])
    with col_dot:
        st.markdown("<h3 style='margin:0; color:#10b981;'>●</h3>", unsafe_allow_html=True)
    with col_status:
        st.markdown(f"**Connected to Ollama**\n`{config['OLLAMA_BASE_URL']}`")

    st.markdown("---")
    st.markdown("### 🛠️ Model Parameters")
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-card-title">Chat Model</div>
            <div class="stat-card-value" style="font-size:1rem;">🤖 {config['OLLAMA_MODEL']}</div>
        </div>
        <div class="stat-card">
            <div class="stat-card-title">Embedding Model</div>
            <div class="stat-card-value" style="font-size:1rem;">🧬 {config['OLLAMA_EMBED_MODEL']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base Index")
    if st.session_state.rag_collection_name:
        st.markdown(
            f"""
            <div class="stat-card" style="border-left: 3px solid #10b981;">
                <div class="stat-card-title">Indexed Documents</div>
                <div class="stat-card-value">{len(st.session_state.indexed_sources)}</div>
                <div class="stat-card-title" style="margin-top:6px;">Total Vectors</div>
                <div class="stat-card-value" style="font-size:1.1rem;">{st.session_state.total_chunks} Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("📄 View Indexed Files", expanded=False):
            for doc in st.session_state.indexed_sources:
                st.markdown(f"- 📄 `{doc}`")
    else:
        st.info("No documents indexed yet. Upload PDFs and index them.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# MAIN AREA: Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
            <div>
                <h1 style="margin:0; font-size:2rem; font-weight:800; letter-spacing:-0.03em;">
                    NLP Chatbot - RAG & NLP Studio
                </h1>
                <p style="margin-top:6px; margin-bottom:0; color:#64748b; font-size:0.95rem;">
                    Local-first document intelligence powered by Ollama, ChromaDB vector indexing, and NLP chunk retrieval.
                </p>
            </div>
            <div>
                <span class="stat-badge">⚡ Privacy-First / Local LLM</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Upload Section & Quick KPI Cards
col_upload, col_summary = st.columns([2, 1])

with col_upload:
    uploaded_files = st.file_uploader(
        "Upload PDF documents to index into local ChromaDB",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents. Scanned image-only PDFs require OCR.",
    )

upload_signature = tuple(
    (file.name, hash(file.getvalue())) for file in (uploaded_files or [])
)

with col_summary:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-card-title">Session State</div>
            <div class="stat-card-value" style="font-size:1.05rem;">
                {'✅ Ready for Queries' if (st.session_state.rag_collection_name and upload_signature == st.session_state.indexed_upload_signature) else '⏳ Indexing Needed'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    index_btn = st.button(
        "⚡ Index & Embed PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

# Handle Indexing Action
if index_btn and uploaded_files:
    with st.status("Processing and indexing documents...", expanded=True) as status:
        try:
            st.write("📖 Extracting text and generating overlapping chunks...")
            chunks = []
            for uploaded_file in uploaded_files:
                chunks.extend(
                    pdf_to_chunks(uploaded_file.getvalue(), uploaded_file.name)
                )

            st.write(
                f"🧬 Computing vector embeddings using `{config['OLLAMA_EMBED_MODEL']}`..."
            )
            client = get_vector_client()
            collection_name = f"pdf_{uuid.uuid4().hex}"
            collection = build_index(
                client,
                collection_name,
                chunks,
                config["OLLAMA_EMBED_MODEL"],
                config["OLLAMA_BASE_URL"],
            )

            previous_collection = st.session_state.rag_collection_name
            if previous_collection:
                client.delete_collection(previous_collection)

            st.session_state.rag_collection_name = collection.name
            st.session_state.indexed_upload_signature = upload_signature
            st.session_state.indexed_sources = [file.name for file in uploaded_files]
            st.session_state.total_chunks = len(chunks)
            st.session_state.messages = []

            status.update(
                label=f"Successfully indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s)!",
                state="complete",
                expanded=False,
            )
            st.toast(f"✅ Indexed {len(chunks)} text chunks successfully!", icon="🎉")
        except (RuntimeError, ValueError) as error:
            status.update(label="Indexing failed", state="error")
            st.error(str(error))

# Warning if files changed since indexing
if st.session_state.rag_collection_name:
    if upload_signature != st.session_state.indexed_upload_signature:
        st.warning(
            "⚠️ Uploaded files have changed since the last index. "
            "Please click **'Index & Embed PDFs'** to refresh your vector index."
        )

# Chat conversation area
st.markdown("---")
st.markdown("### 💬 Document Conversation")

# Sample question helper buttons
if st.session_state.rag_collection_name and not st.session_state.messages:
    st.markdown("<p style='font-size:0.85rem; color:#64748b; margin-bottom:8px;'>Try a quick starter prompt:</p>", unsafe_allow_html=True)
    col_q1, col_q2, col_q3 = st.columns(3)
    preset_prompt = None
    with col_q1:
        if st.button("📋 Executive Summary", use_container_width=True):
            preset_prompt = "Provide a comprehensive executive summary of the indexed documents."
    with col_q2:
        if st.button("🔍 Key Action Items", use_container_width=True):
            preset_prompt = "What are the key requirements, guidelines, or action items outlined?"
    with col_q3:
        if st.button("📊 Quantitative Details", use_container_width=True):
            preset_prompt = "Extract all metrics, numbers, deadlines, or dates mentioned."
else:
    preset_prompt = None

# Render existing messages
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "⚡"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources"):
            sources_html = "".join(
                f"<span class='chat-source-tag'>📄 {src}</span>"
                for src in message["sources"]
            )
            st.markdown(f"<div style='margin-top:6px;'>{sources_html}</div>", unsafe_allow_html=True)

can_query = (
    st.session_state.rag_collection_name is not None
    and upload_signature == st.session_state.indexed_upload_signature
)

user_input = st.chat_input(
    "Ask any question about your indexed PDF documents...",
    disabled=not can_query,
)

active_prompt = preset_prompt if preset_prompt else user_input

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("🔍 Retrieving semantically similar chunks & querying Ollama..."):
            try:
                client = get_vector_client()
                collection = client.get_collection(
                    st.session_state.rag_collection_name
                )
                matches = retrieve_chunks(
                    collection,
                    active_prompt,
                    config["OLLAMA_EMBED_MODEL"],
                    config["OLLAMA_BASE_URL"],
                )
                context = "\n\n".join(
                    f"[{match['source']}, page {match['page']}]\n{match['text']}"
                    for match in matches
                )
                rag_prompt = (
                    "Answer the question using only the supplied PDF context. "
                    "If the context does not contain the answer, say so. "
                    "Cite supporting passages as [filename, page N].\n\n"
                    f"PDF context:\n{context}\n\nQuestion: {active_prompt}"
                )
                conversation = st.session_state.messages[:-1] + [
                    {"role": "user", "content": rag_prompt}
                ]
                reply = ask_ollama(
                    conversation,
                    config["OLLAMA_MODEL"],
                    config["OLLAMA_BASE_URL"],
                )
            except (RuntimeError, ValueError) as error:
                st.error(f"Error: {error}")
            else:
                st.markdown(reply)
                sources = list(
                    dict.fromkeys(
                        f"{match['source']}, page {match['page']}"
                        for match in matches
                    )
                )
                if sources:
                    sources_html = "".join(
                        f"<span class='chat-source-tag'>📄 {src}</span>"
                        for src in sources
                    )
                    st.markdown(
                        f"<div style='margin-top:8px;'><strong>Citations:</strong><br/>{sources_html}</div>",
                        unsafe_allow_html=True,
                    )
                
                # Context inspection expander
                with st.expander("🔍 Inspect Retrieved Context Chunks", expanded=False):
                    for idx, match in enumerate(matches, start=1):
                        st.markdown(f"**Chunk #{idx}** (`{match['source']}`, Page {match['page']})")
                        st.text(match["text"])
                        st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "sources": sources,
                    }
                )
