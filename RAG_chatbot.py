# RAG_chatbot.py - Fixed Version
# ================================

# Import Libraries
import os
import re
import numpy as np
import pdfplumber
from typing import List
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from chromadb import PersistentClient

# -------------------------------
# Config
# -------------------------------
embed_model = "sentence-transformers/all-MiniLM-L6-v2"
chroma_dir = "vector_db"
OPENAI_MODEL = "gpt-3.5-turbo"
MAX_NEW_TOKENS = 300

# Globals
_ce_model = None       # For embeddings
_reranker = None       # For CrossEncoder reranking
_local_model = None
_local_tokenizer = None

# -------------------------------
# PDF Processing
# -------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return re.sub(r"[ \t]+", " ", text).strip()

def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# -------------------------------
# Embeddings
# -------------------------------
def get_embedder():
    global _ce_model
    if _ce_model is None:
        print("Loading embedding model...")
        _ce_model = SentenceTransformer(embed_model)
        print("Embedding model loaded.")
    return _ce_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedder()
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return embs.tolist()

# -------------------------------
# Vector DB
# -------------------------------
def create_vector_db(pdf_path: str):
    client = PersistentClient(path=chroma_dir)
    collection_name = os.path.basename(pdf_path)
    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        # backward compatible
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)
    return client, collection

def ingest_docs(collection, chunks: List[str], embeddings: List[List[float]], pdf_path: str):
    client = PersistentClient(path=chroma_dir)
    if collection.count() > 0:
        print("Document already indexed. Skipping embedding.")
        return
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": i, "source": os.path.basename(pdf_path)} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    try:
        client.persist()
    except Exception:
        pass
    print(f"Ingested {len(chunks)} chunks into vector DB.")

# -------------------------------
# Retrieval + Reranking
# -------------------------------
def retrieve(query: str, collection, top_k: int = 3, source_file=None):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    model = get_embedder()
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()[0]

    candidate_n = max(top_k * 20, 10)
    query_kwargs = dict(query_embeddings=[q_emb], n_results=candidate_n, include=["documents", "distances", "metadatas"])
    if source_file:
        query_kwargs["where"] = {"source": source_file}

    res = collection.query(**query_kwargs)
    docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]

    # Rerank using CrossEncoder
    pairs = [(query, d) for d in docs]
    scores = _reranker.predict(pairs, batch_size=8)
    order = np.argsort(scores)[::-1][:top_k]
    reranked_docs = [docs[i] for i in order]
    reranked_metas = [metas[i] for i in order]
    reranked_dists = [float(1 - (scores[i] - min(scores)) / (max(scores) - min(scores) + 1e-6)) for i in order]

    return {"documents": [reranked_docs], "distances": [reranked_dists], "metadatas": [reranked_metas]}

# -------------------------------
# Prompt / Answer Generation
# -------------------------------
def build_prompt(question: str, context_chunks, max_context_chars=3500):
    context = "\n\n".join(context_chunks)[:max_context_chars]
    prompt = f"""
You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- If answer is not present, say: "I don't know based on the document."
- Be concise (2-4 sentences).
- Do not invent facts.
- Answer using ONLY the information from the context.
- Do NOT copy numbers, tables, or statistics unless needed.
- Summarize the information clearly.
- Answer in 2-3 sentences maximum.

Context:
{context}

Question: {question}

Answer:
"""
    return prompt

def generate_answer_openai(question, context_chunks, user_api_key=None, max_tokens=200):
    prompt = build_prompt(question, context_chunks)
    api_key = user_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key provided.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": "Answer using the document context."},
                  {"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()
    return postprocess_answer(answer)

# Local T5 fallback
def load_local_generator():
    global _local_model, _local_tokenizer
    if _local_model is None:
        model_name = "google/flan-t5-base"
        print("Loading local generator model...")
        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Local model loaded.")
    return _local_tokenizer, _local_model

def generate_answer_local(question, context_chunks, max_tokens=200):
    tokenizer, model = load_local_generator()
    prompt = build_prompt(question, context_chunks)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=4)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return postprocess_answer(answer)

def generate_answer(question, context_chunks, user_api_key=None):
    try:
        if user_api_key or os.getenv("OPENAI_API_KEY"):
            return generate_answer_openai(question, context_chunks, user_api_key)
        else:
            return generate_answer_local(question, context_chunks)
    except Exception:
        print("OpenAI failed, switching to local model.")
        return generate_answer_local(question, context_chunks)

def postprocess_answer(ans: str):
    ans = re.sub(r"\s+", " ", ans)
    ans = re.sub(r"(?i)^(Answer:|Response:)\s*", "", ans)
    return ans.strip()

# -------------------------------
# RAG Pipeline
# -------------------------------
def chatbot(pdf_path_input: str, chunk_size=800, overlap=200):
    client, collection = create_vector_db(pdf_path_input)
    raw_text = extract_text_from_pdf(pdf_path_input)
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_texts(chunks)
    ingest_docs(collection, chunks, embeddings, pdf_path_input)
    return client, collection, chunks

def demo_query(question, collection, source_file=None, top_k=3, use_openai=False, user_api_key=None, verbose=False):
    res = retrieve(question, collection, top_k=top_k, source_file=source_file)
    docs = res["documents"][0]
    if verbose:
        print("Top retrieved chunks:")
        for i, doc in enumerate(docs):
            print(f"{i+1}) {doc[:200]}...\n")
    return generate_answer(question, docs, user_api_key if use_openai else None)

# -------------------------------
# Answer with sources
# -------------------------------
def answer_with_sources(q, top_k=3, source_file=None, use_openai=False, user_api_key=None, collection=None):
    if collection is None:
        import streamlit as st
        collection = st.session_state.get("current_collection")
    if not collection:
        return "No collection loaded.", [], []
    res = retrieve(q, collection=collection, top_k=top_k, source_file=source_file)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    if use_openai:
        ans = generate_answer_openai(q, docs, user_api_key=user_api_key)
    else:
        ans = generate_answer_local(q, docs)
    return ans, docs, metas

# -------------------------------
# Document Summary
# -------------------------------
def generate_document_summary(pdf_path,use_openai=False, user_api_key=None, top_chunks=10, collection=None):
    import streamlit as st
    if collection is None:
        collection = st.session_state.get("current_collection")
    if not collection or "current_pdf" not in st.session_state:
        return "No document loaded."

    source_file = os.path.basename(pdf_path)
    res = collection.get(include=["documents", "metadatas"])
    docs = [d for d, m in zip(res["documents"], res["metadatas"]) if m.get("source") == source_file][:top_chunks]
    if not docs:
        return "No chunks found for the document."

    if use_openai:
        prompt = "Summarize the following document in bullet points:\n\n" + "\n\n".join(docs)
        client = OpenAI(api_key=user_api_key or os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        return postprocess_answer(resp["choices"][0]["message"]["content"])
    else:
        return generate_answer_local("Summarize the document in bullet points.", docs, max_tokens=250)
