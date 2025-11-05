import numpy as np
import pandas as pd
import time
import os
import re
import textwrap
import chromadb
from openai import OpenAI
import pdfplumber
from chromadb import PersistentClient
from typing import List
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# lazy global to avoid repeated loads
_ce_model = None

# Config
pdf_path = "Report_on_AI.pdf"
#embed_model = "sentence-transformers/all-mpnet-base-v2"
#embed_model = "all-MiniLM-L6-v2"
embed_model = "BAAI/bge-large-en"
chroma_dir = "./chroma_report_db"
OPENAI_MODEL = "gpt-3.5-turbo" 
MAX_NEW_TOKENS = 300

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    # Clean noisy spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Chunking Dataset
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Create overlapping chunks of roughly chunk_size characters (or tokens).
    This uses simple char-based windowing which is sufficient for many docs.
    """
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(start + chunk_size, N)
        chunk = text[start:end].strip()
        if end < N:
            next_dot = text.rfind('.', start, end)
            if next_dot > start:
                chunk = text[start:next_dot+1].strip()
                end = next_dot+1
        if chunk:
            chunks.append(chunk)
        prev_start = start
        start = end - overlap
        # Ensure forward progress even if chunk < overlap
        if start <= prev_start:
            start = end
        if len(chunks) > 2000:  # safety
            break
    return [c for c in chunks if len(c) > 50]

# Embeddings
#print("Loading embedding model...")
model = SentenceTransformer(embed_model)
#print("Done")

def embed_texts(texts: List[str]) -> List[List[float]]:
    # Normalize embeddings for cosine similarity and use float32 to save memory
    embs = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return embs.tolist()

# Chroma DB setup and ingest
#print("Setting up Chroma DB....")
client = PersistentClient(path=chroma_dir)
collection_name = "report_kb_bge"
try:
    collection = client.get_or_create_collection(collection_name)
except Exception:
    # Backward-compat: some versions may not have get_or_create_collection
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)
#print("Done....")

def ingest_docs(ids: List[str], docs: List[str], embeddings: List[List[float]], metadatas=None):
    # Avoid duplicate-id errors by deleting if present (no-op if not found)
    try:
        collection.delete(ids=ids)
    except Exception:
        pass
    collection.add(ids=ids, documents=docs, metadatas=metadatas or [{}]*len(ids), embeddings=embeddings)
    # Persist when supported (PersistentClient persists automatically, but this is safe)
    try:
        client.persist()
    except Exception:
        pass
    print(f"Ingested {len(docs)} docs to Chroma at {chroma_dir}")
    
# Retrival function
def retrieve(query: str, top_k: int = 3):
    global _ce_model
    if _ce_model is None:
        _ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")  # free + small

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()[0]
    candidate_n = max(top_k * 20, 60)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=candidate_n,
        include=["documents", "distances", "metadatas"],
        where={"source": os.path.basename(pdf_path)}  # restrict to current file
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    # Cross-encoder rerank
    pairs = [(query, d) for d in docs]
    scores = _ce_model.predict(pairs, batch_size=8)
    order = np.argsort(scores)[::-1][:top_k]  # high -> low
    reranked_docs = [docs[i] for i in order]
    reranked_metas = [metas[i] for i in order]
    reranked_dists = [float(1 - (scores[i] - min(scores)) / (max(scores) - min(scores) + 1e-6)) for i in order]  # pseudo-distance for printing

    return {"documents": [reranked_docs], "distances": [reranked_dists], "metadatas": [reranked_metas]}

# ---------------------------
# 6) Answer generation (Option A: OpenAI Chat)
# ---------------------------
def generate_answer_openai(question: str, context_chunks: List[str], user_api_key=None, max_tokens: int = 300):  # ✅ Added param
    """
    Use OpenAI model to generate answers. If user_api_key is provided, use it; otherwise fallback to env var.
    """
    context = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful assistant that answers questions using the provided context from a report. "
        "If the answer is not in the context, say 'I don't know based on provided document.' "
        "Keep answers concise and factual."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the context above:"

    # ✅ Use user-provided key if available
    if user_api_key:
        OpenAI.api_key = user_api_key
    elif os.getenv("OPENAI_API_KEY"):
        OpenAI.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError("No OpenAI API key provided.")
    client = OpenAI(api_key=user_api_key or os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens
    )
    ans = resp["choices"][0]["message"]["content"].strip()
    return postprocess_answer(ans)

# Answer generation using Hugging Face T5 Local Fallback model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
#gen_model = "google/flan-t5-base"
gen_model = "google/flan-t5-large"
print("Loading HF Generation Model ......")
hf_tokenizer = AutoTokenizer.from_pretrained(gen_model)
hf_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model)
hf_pipe = pipeline("text2text-generation",model = hf_model, max_new_tokens=MAX_NEW_TOKENS, tokenizer = hf_tokenizer, device = -1)

# ---------------------------
# 6b) Answer generation (Option B: HF Flan-T5 local fallback)
# ---------------------------
def generate_answer_hf(question: str, context_chunks: List[str], max_length=200):
    context = "\n\n".join(context_chunks)
    prompt = (
        "Answer using only the given context. Do not invent or generalize. Keep the tone factual and concise (2–3 sentences max)."
        "Use ONLY the provided context to answer the question.\n"
        "If the answer is not explicitly in the context, reply: I don't know based on the provided document.\n"
        "Answer in 2–5 complete sentences. Do not include external links or sources.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    out = hf_pipe(
        prompt,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
        max_new_tokens=max_length,
        min_new_tokens=min(120, max_length),
        length_penalty=1.0,
        truncation=True
    )[0]["generated_text"]
    return postprocess_answer(out)

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
def chatbot(pdf_path_input):
    global pdf_path
    pdf_path = pdf_path_input
    print("Extracting text...")
    raw = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw)} characters.")
    print("Chunking text...")
    chunks = chunk_text(raw, chunk_size=800, overlap=200)
    print(f"Created {len(chunks)} chunks.")
    ids = [f"report_chunk_{i}" for i in range(len(chunks))]
    print("Embedding chunks...")
    embs = embed_texts(chunks)
    print("Ingesting into Chroma...")
    metadatas = [{"source": os.path.basename(pdf_path), "chunk_id": i} for i in range(len(chunks))]
    ingest_docs(ids, chunks, embs, metadatas=metadatas)
    print("Build complete. Now try retrieval & generation.")

def demo_query(q, top_k=2, use_openai=False, user_api_key=None):  # ✅ Added user_api_key param
    res = retrieve(q, top_k=top_k)
    docs = res["documents"][0]
    dists = res["distances"][0]
    print("Top retrieved chunks (distance):")
    for i, (doc, dist) in enumerate(zip(docs, dists)):
        print(f"{i+1}) dist={dist:.4f} -> {doc[:300].replace('\\n',' ')}...\n")

    # ✅ Pass user_api_key to OpenAI if needed
    if use_openai:
        ans = generate_answer_openai(q, docs, user_api_key=user_api_key)
    else:
        ans = generate_answer_hf(q, docs)

    print("\n--- GENERATED ANSWER ---\n")
    print(ans)
    print("\n------------------------\n")
    return ans

#if __name__ == "__main__":
    # Use collection.count() to detect empty store instead of len(collection.get())
#    needs_build = (not os.path.exists(chroma_dir))
#    try:
#        needs_build = needs_build or (collection.count() == 0)
#    except Exception:
        # Fallback for very old versions
#        try:
#            got = collection.get(limit=1)
#           needs_build = needs_build or (len(got.get("ids", [])) == 0)
#        except Exception:
#            needs_build = True

#    if needs_build:
#        chatbot(pdf_path)
#        time.sleep(1)

#    queries = [
#        "What are the main recommendations of the committee on AI platforms ?",
#        "How should personal data be handled according to the report ?",
#        "Does the report talk about algorithmic accountability ?",
#        "What are the index of this pdf ?",
#        "This pdf is about ?"
#    ]
#    for q in queries:
#        print("======== QUERY:", q)
#        demo_query(q, top_k=2, use_openai=False)
