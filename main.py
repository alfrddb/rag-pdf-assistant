import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

load_dotenv()

st.set_page_config(page_title="Advanced RAG Document Assistant", page_icon="📄", layout="wide")

@st.cache_resource
def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please configure it in .env or Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_genai_client()

def extract_and_chunk_pdf(uploaded_file, chunk_size=1000, overlap=200):
    reader = PdfReader(uploaded_file)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "page": page_num})
            start += chunk_size - overlap
    return chunks

def get_embeddings(texts):
    batch_size = 20
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
        )
        for e in response.embeddings:
            val = e.values if hasattr(e, "values") else e["values"]
            all_embeddings.append(val)
    return np.array(all_embeddings, dtype=np.float32)

def hybrid_retrieve_top_k(query, chunks, chunk_embeddings, bm25_index, k=4):
    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    else:
        bm25_scores = np.zeros_like(bm25_scores)

    query_embedding = get_embeddings([query])[0]
    norm_chunks = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norm_query = query_embedding / np.linalg.norm(query_embedding)
    vector_scores = np.dot(norm_chunks, norm_query)
    vector_scores = (vector_scores + 1.0) / 2.0

    hybrid_scores = 0.4 * bm25_scores + 0.6 * vector_scores
    top_indices = np.argsort(hybrid_scores)[::-1][:k]

    return [(chunks[i], float(hybrid_scores[i])) for i in top_indices]

st.title("📄 Advanced Hybrid RAG Assistant")
st.caption("Native Python RAG pipeline featuring Dense Vector Search, BM25 Keyword Search, and Gemini 3.5 Flash-Lite.")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "bm25" not in st.session_state:
    st.session_state.bm25 = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

with st.sidebar:
    st.header("1. Document Management")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.current_file:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Extracting and chunking text...")
            progress_bar.progress(25)
            chunks = extract_and_chunk_pdf(uploaded_file)
            
            if not chunks:
                st.error("Unable to extract text from this PDF.")
            else:
                status_text.text("Building BM25 keyword index...")
                progress_bar.progress(50)
                texts = [c["text"] for c in chunks]
                tokenized_corpus = [text.lower().split() for text in texts]
                bm25_index = BM25Okapi(tokenized_corpus)
                
                status_text.text("Computing semantic embeddings...")
                progress_bar.progress(75)
                embeddings = get_embeddings(texts)
                
                st.session_state.chunks = chunks
                st.session_state.embeddings = embeddings
                st.session_state.bm25 = bm25_index
                st.session_state.current_file = uploaded_file.name
                st.session_state.messages = []
                
                progress_bar.progress(100)
                status_text.empty()
                st.success(f"Successfully indexed {len(chunks)} chunks with Hybrid Search!")

    if st.session_state.chunks:
        st.info(f"Active File: **{st.session_state.current_file}**\n\nTotal Chunks: {len(st.session_state.chunks)}")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Hybrid Retrieval Sources"):
                for src_text, page, score in msg["sources"]:
                    st.markdown(f"**Page {page}** *(Hybrid Score: {score:.3f})*")
                    st.caption(src_text)

if prompt := st.chat_input("Ask a question about the active document..."):
    if st.session_state.embeddings is None:
        st.warning("Please upload a PDF document from the sidebar first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Executing hybrid retrieval and generating response..."):
            retrieved = hybrid_retrieve_top_k(
                prompt, 
                st.session_state.chunks, 
                st.session_state.embeddings, 
                st.session_state.bm25, 
                k=4
            )
            
            context_blocks = [f"[Page {c['page']}]: {c['text']}" for c, _ in retrieved]
            context_str = "\n\n---\n\n".join(context_blocks)

            system_instruction = (
                "You are an expert AI research assistant. Answer the user's question "
                "relying EXCLUSIVELY on the provided CONTEXT. "
                "Always cite the exact page number in parentheses, e.g., (Page 3). "
                "If the answer is not present in the context, explicitly state that the document does not contain this information."
            )

            full_prompt = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{prompt}"

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )

            answer = response.text
            st.markdown(answer)

            sources_data = [(c["text"], c["page"], score) for c, score in retrieved]
            with st.expander("Hybrid Retrieval Sources"):
                for src_text, page, score in sources_data:
                    st.markdown(f"**Page {page}** *(Hybrid Score: {score:.3f})*")
                    st.caption(src_text)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources_data
            })
