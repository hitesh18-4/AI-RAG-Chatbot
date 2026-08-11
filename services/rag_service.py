"""
services/rag_service.py
------------------------
The core business-logic layer of the application.

RAGService is the ONLY class that app.py (the UI) talks to. It orchestrates
the full RAG pipeline by delegating to the specialized modules:

    PDFLoader   -> load & split PDFs into chunks
    ChromaStore -> embed chunks & run semantic search
    prompts     -> build the instruction template for Gemini
    ChatGoogleGenerativeAI -> generate the final answer

This is the layer an interviewer would look at to ask "walk me through how
a question gets answered" — everything UI-related is deliberately kept out
of this file.
"""

from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

from config import Config
from loaders.pdf_loader import PDFLoader
from vectorstore.chroma_store import ChromaStore
from prompts.prompt import get_qa_prompt
from utils.helpers import format_unique_sources


class RAGService:
    """
    Encapsulates the entire RAG workflow as an object with state:
    - which documents have been processed
    - the vector store built from them
    - the QA chain used to answer questions

    Using a class (instead of standalone functions) matters here because the
    service needs to remember state (the built vector store / QA chain)
    across multiple user interactions (upload once, ask many questions) —
    that's a textbook case for OOP over plain functions.
    """

    def __init__(self, google_api_key: str):
        """
        Parameters
        ----------
        google_api_key : str -> Gemini API key, provided by the user via the UI
        """
        self.google_api_key = google_api_key

        # Composition: RAGService "has a" PDFLoader and a ChromaStore rather
        # than inheriting from them. Each collaborator handles one concern.
        self.pdf_loader = PDFLoader()
        self.chroma_store = ChromaStore(google_api_key=google_api_key)

        # These stay None until process_documents() has been called
        self.qa_chain: RetrievalQA | None = None
        self.is_ready: bool = False

    def process_documents(self, uploaded_files: list) -> int:
        """
        Full ingestion pipeline: takes raw uploaded files and prepares the
        system to answer questions about them.

        Steps: save -> load -> split -> embed & store -> build QA chain

        Parameters
        ----------
        uploaded_files : list of Streamlit UploadedFile objects

        Returns
        -------
        int : number of chunks that were created and stored
              (handy for the UI to show "X chunks indexed")
        """
        # Step 1 & 2: Loader + Splitter (delegated to PDFLoader)
        chunks = self.pdf_loader.load_and_split(uploaded_files)

        # Step 3 & 4: Embeddings + ChromaDB (delegated to ChromaStore)
        self.chroma_store.build_from_documents(chunks)

        # Step 5: build the retriever + Gemini QA chain
        self.qa_chain = self._build_qa_chain()
        self.is_ready = True

        return len(chunks)

    def _build_qa_chain(self) -> RetrievalQA:
        """
        Internal helper that wires together the retriever, the Gemini LLM,
        and the prompt template into a single RetrievalQA chain.

        Prefixed with an underscore since this is an implementation detail
        of process_documents() and isn't meant to be called directly from
        outside the class.

        Returns
        -------
        RetrievalQA : chain that returns {"result": ..., "source_documents": [...]}
        """
        retriever = self.chroma_store.get_retriever(k=Config.RETRIEVER_TOP_K)

        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=self.google_api_key,
            temperature=Config.LLM_TEMPERATURE
        )

        prompt = get_qa_prompt()

        # chain_type="stuff" means: take all retrieved chunks and "stuff"
        # them directly into the prompt's {context} slot in one go. This is
        # the simplest RetrievalQA strategy and is appropriate here since we
        # only retrieve a handful of chunks (Config.RETRIEVER_TOP_K) at a time.
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question: str) -> Dict:
        """
        Answer a single question using the previously built QA chain.

        Parameters
        ----------
        question : str -> the user's natural-language question

        Returns
        -------
        Dict with keys:
            "answer"  : str       -> Gemini's generated answer
            "sources" : List[str] -> de-duplicated "<file> - Page <n>" labels

        Raises
        ------
        ValueError if called before process_documents() has been run
        """
        if not self.is_ready or self.qa_chain is None:
            raise ValueError(
                "RAGService is not ready yet. Call process_documents() first."
            )

        # This single call performs the retrieval + generation steps:
        # 1. embeds the question
        # 2. fetches the top-k similar chunks from ChromaDB (semantic search)
        # 3. sends those chunks + question to Gemini using our prompt template
        result = self.qa_chain.invoke({"query": question})

        answer = result["result"]
        source_documents = result["source_documents"]

        # If Gemini says it doesn't know, there's no meaningful context to
        # cite, so we return an empty source list rather than showing
        # possibly-irrelevant chunks next to a "no answer" message.
        if answer.strip() == Config.NO_ANSWER_MESSAGE:
            sources: List[str] = []
        else:
            sources = format_unique_sources(source_documents)

        return {
            "answer": answer,
            "sources": sources
        }

    def reset(self) -> None:
        """
        Clear all state so a fresh set of documents can be processed from
        scratch (e.g. when the user uploads a new batch of PDFs).
        """
        PDFLoader.clear_temp_uploads()
        self.qa_chain = None
        self.is_ready = False
