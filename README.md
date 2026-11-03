# HCL NLP Chatbot

A step-by-step NLP chatbot project. The current chatbot connects to the local
Ollama `llama3.2` model and offers both a terminal interface and a Streamlit
web interface. The terminal version also demonstrates tokenization,
lowercasing, stopword removal, stemming, and lemmatization.

The project roadmap includes punctuation removal, text vectorization, intent
classification, and response generation. The included stemming and
lemmatization examples use small built-in rules for learning purposes.

## Requirements

- Python 3.12 or newer
- [Ollama](https://ollama.com/) installed and running
- The `llama3.2` model downloaded locally

## Setup on Windows

Open PowerShell in the project folder:

```powershell
python -m venv envi
.\envi\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull llama3.2
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
`http://localhost:8501`. Chat history is kept for the current browser session;
use **Clear chat** to start over.

## Run the terminal chatbot

```powershell
python main.py
```

Type `bye`, `exit`, or `quit` to end the terminal conversation.
