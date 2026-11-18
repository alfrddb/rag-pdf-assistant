# 📄 Hybrid RAG Document Assistant

A lightweight, production-oriented Retrieval-Augmented Generation (RAG) application built from scratch in Python. Designed without heavy orchestration frameworks (such as LangChain or LlamaIndex) to demonstrate full control over vector operations, hybrid retrieval mechanics, and hallucination mitigation.

## 🚀 Key Features

* **Hybrid Search Architecture (Dense + Sparse):** Combines semantic similarity via dense vector embeddings (`gemini-embedding-001` + `NumPy` cosine similarity) with exact keyword matching via **BM25** (`rank_bm25`) using a weighted scoring fusion (60% Semantic / 40% Lexical).
* **Custom Sliding-Window Chunking:** Extracts text page-by-page using `pypdf` and applies overlapping character windows (`chunk_size=1000`, `overlap=200`) to preserve cross-boundary semantic context while tracking exact page metadata.
* **Batched API Processing:** Processes large documents efficiently by batching embedding API requests to optimize throughput and respect rate limits.
* **Strict Grounding & Citation Guardrails:** Uses low-temperature generation (`gemini-3.5-flash-lite`) paired with strict system instructions to enforce context-only answers and inline page citations, preventing model hallucinations.
* **Transparent Source Inspection:** Displays the top-$k$ retrieved document chunks alongside their normalized hybrid similarity scores in an expandable UI drawer.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **AI & Embeddings:** Google GenAI SDK (`gemini-3.5-flash-lite`, `gemini-embedding-001`)
* **Retrieval & Math:** `NumPy` (vectorized cosine similarity), `rank-bm25` (BM25Okapi lexical scoring)
* **Document Parsing:** `pypdf`
* **Frontend UI:** `Streamlit`

## 🧠 Architectural Decisions

*  Why no frameworks? Building the chunking, normalization, and dot-product similarity directly with NumPy eliminates abstraction overhead, reduces dependency bloat, and makes the retrieval pipeline transparent and easily     debuggable.

*  Why Hybrid Search? Pure dense vector search often struggles with exact part numbers, rare acronyms, or specific terminology. Fusing normalized BM25 scores with cosine similarity significantly improves recall on             technical and domain-specific PDFs.            
