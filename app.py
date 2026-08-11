"""
app.py
------
Streamlit UI layer — and ONLY the UI layer.

This file should never contain business logic (no PDF parsing, no
embedding calls, no prompt construction). Its only job is to:
    1. Render widgets (sidebar, chat box, chat history)
    2. Read user input
    3. Call RAGService methods
    4. Display whatever RAGService returns

Run with:
    streamlit run app.py
"""

import streamlit as st

from config import Config
from services.rag_service import RAGService
from utils.helpers import get_uploaded_file_names

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG PDF Chat (Gemini)", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDFs (RAG + Gemini + ChromaDB)")
st.caption("Upload PDFs, then ask questions. Answers are grounded only in your documents.")

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
# Streamlit re-runs this whole script top-to-bottom on every interaction.
# session_state is how we "remember" things (the service instance, chat
# history) across those reruns instead of losing them each time.
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None

if "chat_history" not in st.session_state:
    # Each entry: {"question": str, "answer": str, "sources": List[str]}
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ---------------------------------------------------------------------------
# SIDEBAR: API KEY + PDF UPLOAD
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    api_key_input = st.text_input(
        "Gemini API Key",
        value=Config.GEMINI_API_KEY,
        type="password",
        help="Get a free key from https://aistudio.google.com/app/apikey"
    )

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_clicked = st.button("🚀 Process Document(s)", use_container_width=True)

    # --- Handle document processing ---
    if process_clicked:
        if not api_key_input:
            st.error("Please enter your Gemini API key.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Reading PDFs, creating embeddings, and building the index..."):
                # UI creates the service; all real work happens inside it.
                rag_service = RAGService(google_api_key=api_key_input)
                num_chunks = rag_service.process_documents(uploaded_files)

                st.session_state.rag_service = rag_service
                st.session_state.processed_files = get_uploaded_file_names(uploaded_files)
                st.session_state.chat_history = []  # old history belonged to old docs

            st.success(
                f"Processed {len(uploaded_files)} file(s) into {num_chunks} chunks. "
                "Ready to chat!"
            )

    # --- Show which files are currently active ---
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📁 Active documents")
        for name in st.session_state.processed_files:
            st.write(f"- {name}")

    # --- Reset button ---
    if st.session_state.rag_service is not None:
        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True):
            st.session_state.rag_service.reset()
            st.session_state.rag_service = None
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.rerun()

# ---------------------------------------------------------------------------
# MAIN AREA: CHAT
# ---------------------------------------------------------------------------
if st.session_state.rag_service is None:
    st.info("👈 Upload PDF(s) and click **Process Document(s)** in the sidebar to get started.")
else:
    # --- Render existing chat history ---
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry["sources"]:
                with st.expander("📚 Sources"):
                    for src in entry["sources"]:
                        st.write(f"- {src}")

    # --- New question input (pinned at the bottom of the page) ---
    user_question = st.chat_input("Ask a question about your document(s)...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # This is the ONE line where the entire RAG pipeline runs
                # (retrieval + generation) — the UI doesn't need to know how.
                response = st.session_state.rag_service.ask(user_question)

            st.write(response["answer"])
            if response["sources"]:
                with st.expander("📚 Sources"):
                    for src in response["sources"]:
                        st.write(f"- {src}")

        # Persist this exchange so it survives the next rerun
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response["answer"],
            "sources": response["sources"]
        })
