"""
vectorstore/chroma_store.py
----------------------------
Responsible ONLY for embedding chunks and storing/retrieving them from
ChromaDB. This is the "Embeddings -> ChromaDB -> Retriever" part of the
RAG pipeline.

Keeping this isolated means if we ever swapped ChromaDB for FAISS or
Pinecone, only this file would need to change — RAGService, app.py, and
everything else stays untouched.
"""

from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import Config


class ChromaStore:
    """
    Thin wrapper around a Chroma vector store.

    Wrapping it in a class (rather than calling langchain's Chroma directly
    from the service layer) gives us one clear place to:
    - configure the embedding model
    - build a fresh store from new chunks
    - expose a simple `.get_retriever()` method to the rest of the app
    """

    def __init__(self, google_api_key: str):
        self.google_api_key = google_api_key

        # This model converts text into embedding vectors (lists of numbers
        # that capture meaning) using Google's Gemini embedding model.
        # Similar meanings end up close together in vector space, which is
        # what enables "semantic search" instead of plain keyword matching.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=self.google_api_key
        )

        # Holds the actual Chroma instance once it's been built
        self.vectorstore: Chroma | None = None

    def build_from_documents(self, chunks: List[Document]) -> None:
        """
        Embed the given chunks and store them in ChromaDB.

        Parameters
        ----------
        chunks : List[Document] -> text chunks produced by PDFLoader

        Note: we rebuild the store from scratch each time the user processes
        new documents. For a beginner/intermediate project this is simpler
        and more predictable than trying to incrementally update an existing
        persisted store.
        """
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.CHROMA_PERSIST_DIR
        )

    def get_retriever(self, k: int = Config.RETRIEVER_TOP_K):
        """
        Return a retriever object that, given a question, returns the top-k
        most semantically similar chunks from ChromaDB.

        Parameters
        ----------
        k : int -> number of chunks to retrieve per query

        Returns
        -------
        A LangChain retriever (used internally by the QA chain in RAGService)

        Raises
        ------
        ValueError if called before build_from_documents() has populated the store
        """
        if self.vectorstore is None:
            raise ValueError(
                "Vector store has not been built yet. Call build_from_documents() first."
            )

        return self.vectorstore.as_retriever(search_kwargs={"k": k})
