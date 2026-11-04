import streamlit as st
import uuid

from main import ask_ollama, load_config
from rag import build_index, pdf_to_chunks, retrieve_chunks


st.set_page_config(page_title="PDF RAG Chat", page_icon="📄")
st.title("Chat with your PDFs")
st.caption("Upload documents, index them locally, and ask questions with Ollama.")

try:
    config = load_config()
except (OSError, ValueError) as error:
    st.error(f"Could not load chatbot configuration: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_collection_name" not in st.session_state:
    st.session_state.rag_collection_name = None
if "indexed_upload_signature" not in st.session_state:
    st.session_state.indexed_upload_signature = None


@st.cache_resource
def get_vector_client():
    import chromadb

    return chromadb.EphemeralClient()


uploaded_files = st.file_uploader(
    "PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)
upload_signature = tuple(
    (file.name, hash(file.getvalue())) for file in uploaded_files
)

if st.button("Index PDFs", disabled=not uploaded_files):
    with st.spinner("Reading PDFs and creating embeddings..."):
        try:
            chunks = []
            for uploaded_file in uploaded_files:
                chunks.extend(pdf_to_chunks(uploaded_file.getvalue(), uploaded_file.name))

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
            st.session_state.messages = []
            st.success(f"Indexed {len(chunks)} text chunks from {len(uploaded_files)} PDF(s).")
        except (RuntimeError, ValueError) as error:
            st.error(str(error))

if st.session_state.rag_collection_name:
    if upload_signature == st.session_state.indexed_upload_signature:
        st.caption("Indexed: " + ", ".join(st.session_state.indexed_sources))
    else:
        st.warning("The selected PDFs differ from the indexed documents. Select Index PDFs to update the search index.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + "; ".join(message["sources"]))

can_query = (
    st.session_state.rag_collection_name is not None
    and upload_signature == st.session_state.indexed_upload_signature
)
if prompt := st.chat_input("Ask a question about the indexed PDFs", disabled=not can_query):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching PDFs and asking Ollama..."):
            try:
                client = get_vector_client()
                collection = client.get_collection(st.session_state.rag_collection_name)
                matches = retrieve_chunks(
                    collection,
                    prompt,
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
                    f"PDF context:\n{context}\n\nQuestion: {prompt}"
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
                st.error(str(error))
            else:
                st.markdown(reply)
                sources = list(dict.fromkeys(
                    f"{match['source']}, page {match['page']}" for match in matches
                ))
                st.caption("Sources: " + "; ".join(sources))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": sources,
                })

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
