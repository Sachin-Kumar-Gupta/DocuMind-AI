# Importing Libraries
import numpy as np
import pandas as pd
import time
import os
import re
from openai import OpenAI
import torch

# RAG model
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For File reading
from typing import List
import textwrap
import pdfplumber
from PyPDF2 import PdfReader

# For DB management
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings

"""# Config"""

# lazy global to avoid repeated loads
_ce_model = None
_reranker = None

# Embedding model
#embed_model = "sentence-transformers/all-mpnet-base-v2"
#embed_model = "all-MiniLM-L6-v2"
#embed_model = "BAAI/bge-large-en"
embed_model = "sentence-transformers/all-MiniLM-L6-v2"

# Vector database location
chroma_dir = "vector_db"
# Generator model
OPENAI_MODEL = "gpt-3.5-turbo"

# Generation limit
MAX_NEW_TOKENS = 300

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    # Clean excessive spaces but keep line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

# Chunking Dataset
def chunk_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

def get_embedder():
    global _ce_model
    if _ce_model is None:
        print("Loading embedding model...")
        _ce_model = SentenceTransformer(embed_model)
        print("Embedding model loaded.")

    return _ce_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Normalize embeddings for cosine similarity and use float32 to save memory
    model = get_embedder()
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return embs.tolist()

def create_vector_db(pdf_path: str):

    client = PersistentClient(path=chroma_dir)
    collection_name = os.path.basename(pdf_path)
    try:
      collection = client.get_or_create_collection(name=collection_name)
    except Exception:
      # Backward-compat: some versions may not have get_or_create_collection
      try:
        collection = client.get_collection(collection_name)
      except Exception:
        collection = client.create_collection(name=collection_name)
    return client, collection

def ingest_docs(collection, chunks: List[str],embeddings: List[List[float]]):
  client = PersistentClient(path=chroma_dir)

  # Skip if already indexed
  if collection.count() > 0:
    print("Document already indexed. Skipping embedding.")
    return
  ids = [f"chunk_{i}" for i in range(len(chunks))]
  metadatas = [{"chunk_id": i, "source": os.path.basename(pdf_path)} for i in range(len(chunks))]
  collection.add(
      ids=ids,
      documents=chunks,
      embeddings=embeddings,
      metadatas=metadatas
    )
  # Persist when supported (PersistentClient persists automatically, but this is safe)
  try:
    client.persist()
  except Exception:
    pass

  print(f"Ingested {len(chunks)} chunks into vector DB.")

# Retrival function
def retrieve(query: str,collection, top_k: int = 3,source_file=None):
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    model = get_embedder()

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()[0]
    candidate_n = max(top_k * 20, 10)
    if source_file:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=candidate_n,
            include=["documents", "distances", "metadatas"],
            where={"source": source_file}
        )
    else:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=candidate_n,
            include=["documents", "distances", "metadatas"]
        )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    # Cross-encoder rerank
    pairs = [(query, d) for d in docs]
    scores = _reranker.predict(pairs, batch_size=8)
    order = np.argsort(scores)[::-1][:top_k]  # high -> low
    reranked_docs = [docs[i] for i in order]
    reranked_metas = [metas[i] for i in order]
    reranked_dists = [float(1 - (scores[i] - min(scores)) / (max(scores) - min(scores) + 1e-6)) for i in order]  # pseudo-distance for printing

    return {"documents": [reranked_docs], "distances": [reranked_dists], "metadatas": [reranked_metas]}

def build_prompt(question: str, context_chunks, max_context_chars=3500):

    context = "\n\n".join(context_chunks)

    # prevent very long prompts
    context = context[:max_context_chars]

    prompt = f"""
You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- If answer is not present, say: "I don't know based on the document."
- Be concise (2-4 sentences).
- Do not invent facts.

Context:
{context}

Question: {question}

Answer:
"""

    return prompt

# ---------------------------
# 6) Answer generation (Option A: OpenAI Chat)
# ---------------------------
def generate_answer_openai(question, context_chunks, user_api_key=None, max_tokens=200):

    prompt = build_prompt(question, context_chunks)

    api_key = user_api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("No OpenAI API key provided.")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model= OPENAI_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "Answer using the document context."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()

    return postprocess_answer(answer)

# Answer generation using Hugging Face T5 Local Fallback model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

_local_model = None
_local_tokenizer = None

def load_local_generator():

    global _local_model, _local_tokenizer

    if _local_model is None:

        model_name = "google/flan-t5-base"

        print("Loading local generator model...")

        _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _local_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print("Local model loaded.")

    return _local_tokenizer, _local_model

# ---------------------------
# 6b) Answer generation (Option B: HF Flan-T5 local fallback)
# ---------------------------
def generate_answer_local(question, context_chunks, max_tokens=200):

    tokenizer, model = load_local_generator()

    prompt = build_prompt(question, context_chunks)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        num_beams=4
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return postprocess_answer(answer)

def generate_answer(question, context_chunks, user_api_key=None):

    try:

        if user_api_key or os.getenv("OPENAI_API_KEY"):
            return generate_answer_openai(
                question,
                context_chunks,
                user_api_key
            )

        else:
            return generate_answer_local(
                question,
                context_chunks
            )

    except Exception as e:

        print("OpenAI failed, switching to local model.")

        return generate_answer_local(
            question,
            context_chunks
        )

# ========================================================
# 6️⃣ POSTPROCESSING
# ========================================================
def postprocess_answer(ans: str):
    """Clean up model output for readability."""
    ans = re.sub(r"\s+", " ", ans)
    ans = re.sub(r"(?i)^(Answer:|Response:)\s*", "", ans)
    ans = ans.strip()
    return ans

# ---------------------------
# 7) Quick run: extract -> chunk -> embed -> ingest -> test retrieval -> answer
# ---------------------------
def chatbot(pdf_path_input: str, chunk_size: int = 800, overlap: int = 200):
    """
    Full RAG ingestion pipeline: extract -> chunk -> embed -> ingest.
    """
    client, collection = create_vector_db(pdf_path_input)
    raw_text = extract_text_from_pdf(pdf_path_input)
    chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    embeddings = embed_texts(chunks)
    ingest_docs(collection, chunks, embeddings)
    return client, collection, chunks

def demo_query(question, collection, source_file=None, top_k=3, use_openai=False, user_api_key=None, verbose=False):
    """
    Retrieve top-k chunks for a question and generate answer.
    """
    res = retrieve(question, collection, top_k=top_k, source_file=source_file)
    docs = res["documents"][0]
    if verbose:
        print("Top retrieved chunks:")
        for i, doc in enumerate(docs):
            print(f"{i+1}) {doc[:200]}...\n")
    ans = generate_answer(question, docs, user_api_key if use_openai else None)
    return ans

# ========================================================
# 8️ Answer + Sources helper (for UI transparency)
# ========================================================
def answer_with_sources(q, top_k=3, source_file=None, use_openai=False, user_api_key=None, collection=None):
    """
    Returns: answer, retrieved_docs, metadata
    """
    if collection is None:
        collection = st.session_state.get("current_collection")
    if not collection:
        return "No collection loaded.", [], []

    res = retrieve(q, collection=collection, top_k=top_k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    if use_openai:
        ans = generate_answer_openai(q, docs, user_api_key=user_api_key)
    else:
        ans = generate_answer_local(q, docs)

    return ans, docs, metas
# ========================================================
# 9️⃣ Document Summary
# ========================================================
def generate_document_summary(use_openai=False, user_api_key=None, top_chunks=10):
    """Generate bullet-point summary for the current PDF."""

    if "current_pdf" not in st.session_state:
        return "No document loaded."

    source_file = os.path.basename(st.session_state["current_pdf"])

    # Fetch all chunks
    res = collection.get(include=["documents", "metadatas"])

    # Filter chunks for the current PDF
    docs = [
        d for d, m in zip(res["documents"], res["metadatas"])
        if m.get("source") == source_file
    ][:top_chunks]

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
        # Use your local T5 generator
        return generate_answer_local("Summarize the document in bullet points.", docs, max_tokens=250)

'''if __name__ == "__main__":
    # Use collection.count() to detect empty store instead of len(collection.get())
    needs_build = (not os.path.exists(chroma_dir))
    try:
        needs_build = needs_build or (collection.count() == 0)
    except Exception:
        # Fallback for very old versions
        try:
          got = collection.get(limit=1)
          needs_build = needs_build or (len(got.get("ids", [])) == 0)
        except Exception:
          needs_build = True

    if needs_build:
        chatbot(pdf_path)
        time.sleep(1)

    queries = [
        "What are the main recommendations of the committee on AI platforms ?",
        "How should personal data be handled according to the report ?",
        "Does the report talk about algorithmic accountability ?",
        "What are the index of this pdf ?",
        "This pdf is about ?"
    ]
    for q in queries:
        print("======== QUERY:", q)
        demo_query(q, top_k=2, use_openai=False)'''
