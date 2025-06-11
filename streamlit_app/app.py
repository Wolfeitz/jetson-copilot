import ollama
import openai
import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from PIL import Image
import time
import logging
import sys
import io
import utils.func 
import utils.constants as const

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

st.set_page_config(page_title="Jetson Copilot", menu_items=None)

AVATAR_AI   = Image.open('./images/jetson-soc.png')
AVATAR_USER = Image.open('./images/user-purple.png')

DEFAULT_PROMPT = """You are a chatbot, able to have normal interactions, as well as talk about NVIDIA Jetson embedded AI computer.
Here are the relevant documents for the context:\n
{context_str}
\nInstruction: Use the previous chat history, or the context above, to interact and help the user."""

# --- Session State Initialization ---
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question about NVIDIA Jetson embedded AI computer!", "avatar": AVATAR_AI}
    ]

if "context_prompt" not in st.session_state:
    st.session_state.context_prompt = DEFAULT_PROMPT

if "use_index" not in st.session_state:
    st.session_state.use_index = False

# --- Utility Functions ---
def find_saved_indexes():
    return utils.func.list_directories(const.INDEX_ROOT_PATH)

def load_index(index_name):
    Settings.embed_model = OllamaEmbedding("mxbai-embed-large:latest")
    dir = f"{const.INDEX_ROOT_PATH}/{index_name}"
    storage_context = StorageContext.from_defaults(persist_dir=dir)
    return load_index_from_storage(storage_context)

# --- Model Setup ---
models = [model["name"] for model in ollama.list()["models"]]

if 'llama3:latest' not in models:
    with st.spinner('Downloading llama3 model ...'):
        ollama.pull('llama3')

if 'mxbai-embed-large:latest' not in models:
    with st.spinner('Downloading mxbai-embed-large model ...'):
        ollama.pull('mxbai-embed-large')

# --- Sidebar ---
with st.sidebar:
    st.title(":airplane: Jetson Copilot")
    st.subheader('Your local AI assistant on Jetson', divider='rainbow')

    st.session_state["model"] = st.selectbox("Choose your LLM", models, index=models.index("llama3:latest"))
    Settings.llm = Ollama(model=st.session_state["model"], request_timeout=300.0)
    st.page_link("pages/download_model.py", label=" Download a new LLM", icon="➕")

    # Toggle RAG
    prev_use_index = st.session_state.use_index
    st.session_state.use_index = st.toggle("Use RAG (adds context from documents)", value=st.session_state.use_index)

    # Prompt editor
    new_prompt = st.text_area("System prompt", st.session_state.context_prompt, height=240)
    prompt_changed = new_prompt != st.session_state.context_prompt
    if prompt_changed:
        st.session_state.context_prompt = new_prompt
        st.success("System prompt updated!")

    # Load index and initialize chat engine if RAG is enabled
    if st.session_state.use_index:
        saved_index_list = find_saved_indexes()
        index_name = st.selectbox("Index", saved_index_list)

        if index_name:
            with st.spinner('Loading Index...'):
                st.session_state.index = load_index(index_name)

            st.page_link("pages/build_index.py", label=" Build a new index", icon="➕")

            # Auto-init chat engine when toggled or prompt changed
            if prompt_changed or not prev_use_index or "chat_engine" not in st.session_state:
                st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                    chat_mode="context",
                    streaming=True,
                    memory=st.session_state.memory,
                    llm=Settings.llm,
                    context_prompt=st.session_state.context_prompt,
                    verbose=True
                )
                st.success("RAG engine initialized!")

    # Reset conversation
    if st.button("🔄 Reset Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me any question about NVIDIA Jetson embedded AI computer!", "avatar": AVATAR_AI}
        ]
        st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        st.success("Chat history reset.")

    # Export transcript
    if st.button("📄 Export Transcript"):
        transcript = "\n\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages if m['role'] in ['user', 'assistant']
        )
        buffer = io.BytesIO()
        buffer.write(transcript.encode())
        buffer.seek(0)
        st.download_button("Download Chat (.txt)", data=buffer, file_name="chat_transcript.txt", mime="text/plain")

# --- Chat function ---
def model_res_generator(prompt=""):
    system_prompt = st.session_state.context_prompt

    if st.session_state.use_index and "chat_engine" in st.session_state:
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        for chunk in response_stream.response_gen:
            yield chunk
    else:
        messages_only = [{"role": "system", "content": system_prompt}]
        messages_only += [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        stream = ollama.chat(model=st.session_state["model"], messages=messages_only, stream=True)
        for chunk in stream:
            yield chunk["message"]["content"]

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])

# --- Handle user input ---
if prompt := st.chat_input("Enter prompt here.."): 
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": AVATAR_USER})

    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            time.sleep(1)
            message = st.write_stream(model_res_generator(prompt))
            st.session_state.messages.append({"role": "assistant", "content": message, "avatar": AVATAR_AI})
