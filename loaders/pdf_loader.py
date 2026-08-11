"""
loaders/pdf_loader.py
----------------------
Responsible ONLY for turning uploaded PDF files into clean, chunked
LangChain Document objects. Nothing about embeddings, ChromaDB, or Gemini
belongs here — that separation is what makes the project "clean architecture".

This module covers the first two RAG pipeline steps:
    Loader (PyPDFLoader) -> Splitter (RecursiveCharacterTextSplitter)
"""

import os
import shutil
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import Config


class PDFLoader:
    """
    Handles saving uploaded PDFs to disk, reading them, and splitting them
    into small overlapping chunks ready for embedding.

    Wrapping this in a class (instead of loose functions) makes it easy to
    configure chunk_size/chunk_overlap once per instance and reuse it,
    and it mirrors how a real codebase would let you swap in a different
    loader (e.g. DocxLoader) later without touching the rest of the app.
    """

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries to split on natural boundaries
        # first (paragraphs -> sentences -> words) before falling back to a
        # hard character cut, which keeps chunks more semantically coherent.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def save_uploaded_files(self, uploaded_files: list) -> List[str]:
        """
        Streamlit's file_uploader gives us in-memory file objects, but
        PyPDFLoader needs real file paths on disk. This saves each uploaded
        file into a temp folder and returns the list of saved paths.

        Parameters
        ----------
        uploaded_files : list of Streamlit UploadedFile objects

        Returns
        -------
        List[str] : file paths where the PDFs were saved
        """
        os.makedirs(Config.UPLOAD_TEMP_DIR, exist_ok=True)

        saved_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(Config.UPLOAD_TEMP_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            saved_paths.append(file_path)

        return saved_paths

    def load_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        Load one or more PDFs from disk into LangChain Document objects.

        PyPDFLoader creates one Document per page, and automatically attaches
        metadata such as {"source": file_path, "page": page_number}. That
        metadata is exactly what lets us later show "which document/page did
        this answer come from".

        Parameters
        ----------
        pdf_paths : List[str] -> paths returned by save_uploaded_files()

        Returns
        -------
        List[Document] : one Document per PDF page, across all given PDFs
        """
        all_documents: List[Document] = []

        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load()
            all_documents.extend(pages)

        return all_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split full-page documents into smaller overlapping chunks.

        Why split further even though PyPDFLoader already gives per-page
        documents? Pages can still be too large (or too small/fragmented)
        for good embedding quality, so we normalize everything into
        consistently-sized chunks.

        Parameters
        ----------
        documents : List[Document] -> output of load_documents()

        Returns
        -------
        List[Document] : smaller chunks, each still carrying the original
                          page's metadata (source file + page number)
        """
        return self.splitter.split_documents(documents)

    def load_and_split(self, uploaded_files: list) -> List[Document]:
        """
        Convenience method that runs the full loader pipeline in one call:
        save -> load -> split.

        This is the single method the service layer (RAGService) will call,
        so it doesn't need to know about the individual steps.

        Parameters
        ----------
        uploaded_files : list of Streamlit UploadedFile objects

        Returns
        -------
        List[Document] : final chunks ready to be embedded
        """
        pdf_paths = self.save_uploaded_files(uploaded_files)
        documents = self.load_documents(pdf_paths)
        chunks = self.split_documents(documents)
        return chunks

    @staticmethod
    def clear_temp_uploads() -> None:
        """
        Removes the temporary upload folder entirely.

        Useful when a user uploads a fresh set of PDFs and we don't want
        old files lingering around and being accidentally re-read.
        """
        if os.path.exists(Config.UPLOAD_TEMP_DIR):
            shutil.rmtree(Config.UPLOAD_TEMP_DIR)
