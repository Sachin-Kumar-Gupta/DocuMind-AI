# 🤖 DocuMind AI — RAG-Based PDF Q&A Assistant

**DocuMind AI** is an intelligent **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload any PDF and ask natural language questions about it.  
It uses **Sentence Transformers**, **ChromaDB**, and **Streamlit** to create a smart, local-first assistant — with an optional **OpenAI GPT mode** for enhanced accuracy and fluency.

---

## 🌟 Features

✅ Upload any **PDF document**  
✅ Ask **natural language questions** about its content  
✅ Choose between two modes:
- **💎 OpenAI GPT Mode:** High accuracy & fluent responses  
- **⚙️ Local Mode (Flan-T5):** 100% free & offline  

✅ Clean, interactive **Streamlit UI**  
✅ Stores document embeddings using **ChromaDB**  
✅ Built entirely in **Python** — no heavy setup required  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| Vector Database | 🟣 ChromaDB |
| Embedding Model | 🧩 `BAAI/bge-large-en` (Sentence Transformers) |
| Reranker | ⚡ `cross-encoder/ms-marco-MiniLM-L6-v2` |
| LLM (Local) | 🏠 Flan-T5 Large |
| LLM (Cloud Option) | ☁️ OpenAI GPT-3.5/4 |
| Frontend | 💻 Streamlit |
| PDF Reader | 📄 pdfplumber |

---
