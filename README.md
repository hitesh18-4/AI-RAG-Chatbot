# 📄 RAG PDF Chat — Clean Architecture Edition

A beginner-to-intermediate **Retrieval-Augmented Generation (RAG)** project built with a
clean, layered architecture — designed to be walked through in a software developer
interview.

Upload multiple PDFs, chat with them, and get answers grounded **only** in the uploaded
content, with the exact source document and page number shown for every answer.

## 🧠 RAG Pipeline

```
PDF Upload
   │
   ▼
PyPDFLoader  ───────────►  loads each PDF into per-page Documents
   │
   ▼
RecursiveCharacterTextSplitter ──►  splits pages into small overlapping chunks
   │
   ▼
GoogleGenerativeAIEmbeddings ───►  converts each chunk into a vector
   │
   ▼
ChromaDB  ──────────────────────►  stores vectors for semantic search
   │
   ▼
Retriever ──────────────────────►  finds top-k chunks most similar to the question
   │
   ▼
Gemini (via prompt template) ───►  answers using ONLY the retrieved chunks
```

## 🗂️ Folder structure

```
project/
│
├── app.py                     # Streamlit UI ONLY — no business logic
├── config.py                  # All settings/constants in one place
│
├── services/
│   └── rag_service.py         # RAGService class — orchestrates the whole pipeline
│
├── loaders/
│   └── pdf_loader.py          # PDFLoader class — load & split PDFs
│
├── vectorstore/
│   └── chroma_store.py        # ChromaStore class — embeddings + ChromaDB
│
├── prompts/
│   └── prompt.py              # Prompt template (anti-hallucination instructions)
│
├── utils/
│   └── helpers.py             # Small stateless helper functions
│
├── requirements.txt
└── README.md
```

Each folder has exactly one job. This is the same "separation of concerns" idea used
in larger backend systems (controllers vs. services vs. repositories), just scaled
down to a small project.

## ⚙️ Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get a free Gemini API key from https://aistudio.google.com/app/apikey

4. (Optional) Save it in a `.env` file in the project root so you don't have to
   paste it every time:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## ▶️ Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` by default.

## 💡 Usage

1. Enter your Gemini API key (or let it load from `.env`).
2. Upload one or more PDFs in the sidebar.
3. Click **Process Document(s)** — this builds the ChromaDB index.
4. Ask questions in the chat box.
5. Expand **📚 Sources** under any answer to see which file/page it came from.
6. Use **Clear session** to reset and upload a different set of documents.

## 🧩 Design decisions (for interview discussion)

- **Why split into `loaders/`, `vectorstore/`, `prompts/`, `services/`?**
  Each module has a single, clear responsibility. If we swapped ChromaDB for FAISS,
  only `chroma_store.py` would change. If we tweaked the prompt wording, only
  `prompt.py` would change. `app.py` and `rag_service.py` wouldn't need to know.

- **Why a `RAGService` class instead of plain functions?**
  The service needs to hold state across multiple user interactions — you upload
  documents once, then ask many questions against the same vector store and QA
  chain. A class naturally models "build once, use many times" via instance
  attributes (`self.qa_chain`, `self.is_ready`). It also uses **composition**
  (`self.pdf_loader = PDFLoader()`, `self.chroma_store = ChromaStore(...)`) rather
  than inheritance, since `RAGService` *uses* a loader and a vector store — it
  isn't *a kind of* either one.

- **Why does `app.py` only call `rag_service.process_documents()` and
  `rag_service.ask()`?**
  This is the "thin controller" pattern — the UI layer should be replaceable. If
  we swapped Streamlit for a CLI or a different frontend, `RAGService` wouldn't
  need to change at all.

- **Why centralize the prompt in `prompts/prompt.py`?**
  The prompt is what enforces "answer only from context" and the exact fallback
  message. Keeping it separate from the chain-building logic in `rag_service.py`
  makes the anti-hallucination behavior easy to find and easy to tune independently.

- **Why `chain_type="stuff"`?**
  It's the simplest RetrievalQA strategy — all retrieved chunks get "stuffed"
  directly into one prompt. Appropriate here since we only retrieve a handful of
  chunks (`Config.RETRIEVER_TOP_K`) at a time. Other strategies (`map_reduce`,
  `refine`) exist for when the retrieved content is too large to fit in one prompt.

- **Why rebuild the ChromaDB index from scratch each time instead of updating it
  incrementally?**
  Simpler and more predictable for a project this size. Incremental updates
  (checking for duplicate chunks, deletions, etc.) add real complexity that isn't
  needed to demonstrate the core RAG concept.

- **What was deliberately left out, and why?**
  No FastAPI, Docker, authentication, async, Redis, or Celery — none of that is
  necessary to demonstrate RAG fundamentals or clean architecture, and adding it
  would obscure the core logic an interviewer actually wants to see.

## 🚀 Natural extensions (good talking points, not implemented)

- Persist the processed-document list across app restarts (currently in-memory only).
- Add unit tests for `PDFLoader` and `ChromaStore` using small sample PDFs.
- Show similarity scores next to retrieved sources.
- Support additional file types (`.docx`, `.txt`) by adding a new loader class
  alongside `PDFLoader`.
