import streamlit as st

from main import ask_ollama, load_config


st.set_page_config(page_title="Llama 3.2 Chatbot", page_icon="💬")
st.title("Llama 3.2 Chatbot")
st.caption("Chat with your local Ollama model.")

try:
    config = load_config()
except (OSError, ValueError) as error:
    st.error(f"Could not load chatbot configuration: {error}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = ask_ollama(
                    st.session_state.messages,
                    config["OLLAMA_MODEL"],
                    config["OLLAMA_BASE_URL"],
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.markdown(reply)
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )

if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
