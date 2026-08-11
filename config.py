"""
config.py
---------
Centralized configuration for the whole project.

Why this file exists:
- Interviewers like to see that "magic values" (model names, chunk sizes,
  folder paths, etc.) are not scattered across the codebase.
- If we ever need to change the embedding model, chunk size, or the number
  of retrieved chunks, we change it in exactly ONE place.

This module does not contain any business logic — only constants and
simple environment loading.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
# Example .env content:
#   GEMINI_API_KEY=your_key_here
load_dotenv()


class Config:
    """
    Groups all configuration values as class attributes.

    Using a class (instead of loose module-level variables) makes it easy to
    import as `from config import Config` and access everything via
    `Config.SOMETHING`, which reads clearly at call sites.
    """

    # ---------------- API KEY ----------------
    # Falls back to an empty string if not set; app.py will ask the user
    # to enter it manually in the sidebar in that case.
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # ---------------- MODELS ----------------
    # Gemini chat model used to generate the final answer.
    # Use a current alias to avoid retired model names.
    LLM_MODEL: str = "gemini-flash-latest"

    # Gemini embedding model used to convert text chunks into vectors.
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # temperature=0 keeps answers focused and factual rather than "creative",
    # which matters for a RAG system that should stick to the given context.
    LLM_TEMPERATURE: float = 0.0

    # ---------------- TEXT SPLITTING ----------------
    # Max characters per chunk before splitting
    CHUNK_SIZE: int = 1000

    # Overlap between consecutive chunks so context isn't lost at chunk edges
    CHUNK_OVERLAP: int = 150

    # ---------------- RETRIEVAL ----------------
    # Number of most-relevant chunks to fetch from ChromaDB per question
    RETRIEVER_TOP_K: int = 4

    # ---------------- STORAGE ----------------
    # Folder where ChromaDB persists its vector data on disk
    CHROMA_PERSIST_DIR: str = "chroma_db"

    # Folder where uploaded PDFs are temporarily saved before being read
    UPLOAD_TEMP_DIR: str = "temp_uploads"

    # ---------------- FALLBACK MESSAGE ----------------
    # Returned when the retrieved context does not contain a relevant answer.
    # Centralizing this string means the UI and the service always stay in sync
    # if we ever want to reword it.
    NO_ANSWER_MESSAGE: str = "I don't have enough information from the uploaded documents."