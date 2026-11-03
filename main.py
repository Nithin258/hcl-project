import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "with",
}

IRREGULAR_LEMMAS = {
    "am": "be",
    "are": "be",
    "children": "child",
    "did": "do",
    "does": "do",
    "had": "have",
    "has": "have",
    "mice": "mouse",
    "ran": "run",
    "went": "go",
    "were": "be",
    "was": "be",
}


def tokenize(text):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text)


def stem_word(word):
    """Apply a few common suffix rules; this is not a full Porter stemmer."""
    if len(word) > 5 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def lemmatize_word(word):
    """Normalize common inflections using small rules and a tiny word map."""
    if word in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        base = word[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(word) > 5 and word.endswith("ily"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ed"):
        base = word[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        if base.endswith(("v", "k")):
            base += "e"
        return base
    if len(word) > 4 and word.endswith("ly"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def load_config():
    config = {
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
    }
    env_path = Path(__file__).with_name(".env")

    if env_path.exists():
        for line_number, line in enumerate(
            env_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError(f"Invalid .env entry on line {line_number}.")
            key = key.strip()
            if key in config:
                config[key] = value.strip().strip("\"'")

    for key in config:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def ask_ollama(messages, model, base_url):
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(
            {"model": model, "messages": messages, "stream": False}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace").strip()
        message = f"Ollama returned HTTP {error.code}"
        if details:
            message += f": {details}"
        raise RuntimeError(message) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            f"Make sure Ollama is running. Details: {error.reason}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama returned an invalid JSON response.") from error

    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        raise RuntimeError(f"Ollama error: {payload['error']}")
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("message"), dict)
        or not isinstance(payload["message"].get("content"), str)
    ):
        raise RuntimeError("Ollama returned a response in an unexpected format.")

    return payload["message"]["content"]


def main():
    config = load_config()
    conversation = []
    print("Bot: Hello! Chat with me, or type 'bye' to exit.")

    while True:
        try:
            message = input("You: ")
        except EOFError:
            break

        tokens = tokenize(message)
        lowercase_tokens = [token.lower() for token in tokens]
        filtered_tokens = [
            token for token in lowercase_tokens if token not in STOPWORDS
        ]
        stemmed_tokens = [stem_word(token) for token in filtered_tokens]
        lemmatized_tokens = [
            lemmatize_word(token) for token in filtered_tokens
        ]

        print(f"1. Tokenization: {tokens}")
        print(f"2. Lowercase conversion: {lowercase_tokens}")
        print(f"3. Stopword removal: {filtered_tokens}")
        print(f"4. Stemming: {stemmed_tokens}")
        print(f"5. Lemmatization: {lemmatized_tokens}")

        conversation.append({"role": "user", "content": message})
        try:
            response = ask_ollama(
                conversation,
                config["OLLAMA_MODEL"],
                config["OLLAMA_BASE_URL"],
            )
        except RuntimeError as error:
            conversation.pop()
            print(f"Bot: Error: {error}")
            continue

        conversation.append({"role": "assistant", "content": response})
        print(f"Bot: {response}")

        if message.strip().lower() in {"bye", "exit", "quit"}:
            break


if __name__ == "__main__":
    main()
