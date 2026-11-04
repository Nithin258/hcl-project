# HCL NLP Chatbot

A step-by-step NLP chatbot project. The terminal chatbot connects to Ollama and
demonstrates tokenization, lowercasing, stopword removal, stemming, and
lemmatization. The Streamlit app lets you upload PDFs, index their text in a
local Chroma vector database using Ollama embeddings, and ask questions against
the retrieved passages.

The project roadmap includes punctuation removal, text vectorization, intent
classification, and response generation. The included stemming and
lemmatization examples use small built-in rules for learning purposes.

## Requirements

- Python 3.12 or newer
- [Ollama](https://ollama.com/) installed and running
- The `llama3.2` model downloaded locally
- The `nomic-embed-text` model downloaded locally for PDF embeddings

## Setup on Windows

Open PowerShell in the project folder:

```powershell
python -m venv envi
.\envi\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
ollama pull nomic-embed-text
```

Set the Ollama configuration in a local `.env` file:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

The `.env` file is ignored by Git. Environment variables with the same names
take precedence.

## Run the Streamlit app

Make sure Ollama is running, then run:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually
`http://localhost:8501`. Upload one or more text-based PDFs and select **Index
PDFs** before asking a question. The vector index is held in memory for the
current app session. Scanned PDFs require OCR before they can be searched.
Chat history is kept for the current browser session; use **Clear chat** to
start over.

## Run the terminal chatbot

```powershell
python main.py
```

Type `bye`, `exit`, or `quit` to end the terminal conversation.
