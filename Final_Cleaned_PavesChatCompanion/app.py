import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
from datetime import datetime

from rag_system import RAGSystem
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_client import LLMClient
from utils import setup_logging, format_sources, clean_text
from dotenv import load_dotenv
# ⚠️ Security Warning
# Replace this with environment variable usage in production
load_dotenv()  # Loads variables from .env file into environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

logger = setup_logging()

st.set_page_config(
    page_title="Paves Technologies - AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1f4e79;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        color: #4a5568;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    .company-logo {
        text-align: center;
        color: #1f4e79;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = []

def initialize_rag_system():
    try:
        groq_api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            st.error("⚠️ GROQ_API_KEY is missing. Please set it.")
            return None

        document_processor = DocumentProcessor()
        vector_store = VectorStore()
        llm_client = LLMClient(api_key=groq_api_key)

        rag_system = RAGSystem(
            document_processor=document_processor,
            vector_store=vector_store,
            llm_client=llm_client,
        )
        return rag_system

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return

    if st.session_state.rag_system is None:
        st.session_state.rag_system = initialize_rag_system()
        if st.session_state.rag_system is None:
            return

    progress_bar = st.progress(0)
    status_text = st.empty()

    processed_count = 0
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name} …")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            ok = st.session_state.rag_system.add_document(
                file_path=tmp_path,
                filename=uploaded_file.name,
            )

            if ok:
                st.session_state.documents_processed.append(uploaded_file.name)
                processed_count += 1
                st.success(f"✅ {uploaded_file.name} processed")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")

            os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {e}")
            st.error(f"Error processing {uploaded_file.name}: {e}")

        progress_bar.progress((i + 1) / total_files)

    status_text.text(f"Done – processed {processed_count}/{total_files} PDFs")
    # st.balloons() has been removed

def display_chat_messages():
    for msg in st.session_state.get("messages", []):
        if not isinstance(msg, dict):
            continue
        with st.chat_message(msg.get("role", "user")):
            st.markdown(msg.get("content", ""))
            if msg.get("sources"):
                with st.expander("📄 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['filename']}** (Page {src.get('page','N/A')})")
                        st.markdown(f"*Relevance: {src['score']:.2f}*")
                        st.markdown(f"> {src['content'][:200]}…")
                        st.markdown("---")

def main():
    st.markdown('<div class="company-logo"> PAVES TECHNOLOGIES</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">AI‑Powered Document Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Get instant answers from your company documents using advanced AI</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("📁 Document Management")
        uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            if st.button("🔄 Process Documents", type="primary"):
                process_uploaded_files(uploaded_files)

        if st.session_state.documents_processed:
            st.subheader("📋 Processed Documents")
            for doc in st.session_state.documents_processed:
                st.write(f"✅ {doc}")

        st.subheader("🔧 System Status")
        if st.session_state.rag_system:
            st.write(f"📄 Documents: {len(st.session_state.documents_processed)}")
            st.write("🟢 System: Ready")
        else:
            st.write("🔴 System: Not initialized")

        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()

    st.header("💬 Chat with Your Documents")
    display_chat_messages()

    if prompt := st.chat_input("Ask me anything about your documents…"):
        if not st.session_state.rag_system:
            st.session_state.rag_system = initialize_rag_system()
            if st.session_state.rag_system is None:
                return

        if not st.session_state.documents_processed:
            st.warning("⚠️ Upload and process documents first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    data = st.session_state.rag_system.query(prompt)
                    if data:
                        placeholder = st.empty()
                        answer_words = data["answer"].split()
                        acc = ""
                        for w in answer_words:
                            acc += w + " "
                            placeholder.markdown(acc + "▌")
                        placeholder.markdown(acc)

                        if data.get("sources"):
                            with st.expander("📄 Sources", expanded=False):
                                for src in data["sources"]:
                                    st.markdown(f"**{src['filename']}** (Page {src.get('page','N/A')})")
                                    st.markdown(f"*Relevance: {src['score']:.2f}*")
                                    st.markdown(f"> {src['content'][:200]}…")
                                    st.markdown("---")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": acc,
                            "sources": data.get("sources", []),
                        })
                    else:
                        err = "I'm sorry, I couldn't find relevant information to answer your question."
                        st.markdown(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
                except Exception as e:
                    err = f"An error occurred while processing your question: {e}"
                    logger.error(err)
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": "I encountered an error. Please try again."})

if __name__ == "__main__":
    main()
