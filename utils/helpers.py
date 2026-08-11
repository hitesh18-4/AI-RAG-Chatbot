"""
utils/helpers.py
-----------------
Small, generic helper functions that don't belong to any single layer
(loader, vectorstore, or service). Kept separate so those modules stay
focused on their one responsibility.
"""

import os
from typing import List
from langchain_core.documents import Document


def format_source(doc: Document) -> str:
    """
    Build a short, human-readable citation string from a retrieved chunk's
    metadata, e.g. "resume.pdf - Page 2".

    Parameters
    ----------
    doc : Document -> a chunk returned by the retriever, with metadata
                       attached by PyPDFLoader (source path + page number)

    Returns
    -------
    str : formatted "<filename> - Page <n>" label
    """
    source_path = doc.metadata.get("source", "Unknown source")
    file_name = os.path.basename(source_path)

    # PyPDFLoader's page numbers are 0-indexed internally, so add 1 to
    # display a natural, human-friendly page number.
    page_number = doc.metadata.get("page")
    if page_number is not None:
        return f"{file_name} - Page {page_number + 1}"
    return file_name


def format_unique_sources(source_documents: List[Document]) -> List[str]:
    """
    Convert a list of retrieved chunks into a de-duplicated list of
    formatted source labels, preserving the original order.

    Multiple chunks can come from the same page (or the same source can
    appear more than once), so we remove duplicates before displaying them.

    Parameters
    ----------
    source_documents : List[Document] -> chunks returned by the QA chain

    Returns
    -------
    List[str] : unique, ordered "<filename> - Page <n>" labels
    """
    sources = [format_source(doc) for doc in source_documents]

    # dict.fromkeys() preserves insertion order while dropping duplicates,
    # which a plain set() would not guarantee.
    return list(dict.fromkeys(sources))


def get_uploaded_file_names(uploaded_files: list) -> List[str]:
    """
    Extract just the file names from a list of Streamlit UploadedFile
    objects, mainly used for displaying a short "Processed: a.pdf, b.pdf"
    style confirmation message in the UI.
    """
    return [f.name for f in uploaded_files]
