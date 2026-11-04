import json
import uuid
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader


def pdf_to_chunks(pdf_bytes, source_name, chunk_size=1000, chunk_overlap=150):
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted and not reader.decrypt(""):
            raise RuntimeError(f"{source_name} is password-protected and cannot be read.")
        chunks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "id": uuid.uuid4().hex,
                        "text": chunk_text,
                        "metadata": {"source": source_name, "page": page_number},
                    })
                if end == len(text):
                    break
                start = end - chunk_overlap
        if not chunks:
            raise RuntimeError(
                f"No extractable text was found in {source_name}. "
                "Scanned PDFs need OCR before they can be searched."
            )
        return chunks
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"Could not read {source_name}: {error}") from error


def _ollama_embeddings(texts, model, base_url):
    request = Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        message = f"Ollama returned HTTP {error.code} while creating embeddings"
        if details:
            message += f": {details}"
        raise RuntimeError(message) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. Make sure Ollama is running."
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama returned an invalid embedding response.") from error

    embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
    if (
        not isinstance(embeddings, list)
        or len(embeddings) != len(texts)
        or any(not isinstance(vector, list) or not vector for vector in embeddings)
    ):
        error = payload.get("error") if isinstance(payload, dict) else None
        if error:
            raise RuntimeError(f"Ollama embedding error: {error}")
        raise RuntimeError("Ollama returned embeddings in an unexpected format.")
    return embeddings


def build_index(client, collection_name, chunks, embedding_model, base_url):
    if not chunks:
        raise ValueError("Upload at least one PDF containing searchable text.")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    try:
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            embeddings = _ollama_embeddings(
                [chunk["text"] for chunk in batch], embedding_model, base_url
            )
            collection.add(
                ids=[chunk["id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
                embeddings=embeddings,
            )
    except Exception:
        client.delete_collection(collection_name)
        raise
    return collection


def retrieve_chunks(collection, query, embedding_model, base_url, top_k=4):
    count = collection.count()
    if count == 0:
        raise ValueError("The PDF search index is empty. Index the PDFs again.")
    query_embedding = _ollama_embeddings([query], embedding_model, base_url)[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [
        {
            "text": document,
            "source": metadata.get("source", "Unknown PDF"),
            "page": metadata.get("page", "?"),
        }
        for document, metadata in zip(documents, metadatas)
    ]