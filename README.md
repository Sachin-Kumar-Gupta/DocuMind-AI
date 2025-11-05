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
## 🧩 Example Workflow

1. Upload your PDF

2. Choose OpenAI mode (enter your API key) or Local mode

3. Ask questions like:
    - What are the key recommendations in this report?
    - Does the document discuss data privacy?

4. Get accurate, context-based answers 🎯


## 📦 Project Structure

📁 DocuMind-AI/

│
├── app.py                   # Streamlit UI

├── RAG_chatbot.py           # Core RAG pipeline

├── requirements.txt          # Dependencies

├── README.md                 # Documentation

├── .gitignore                # Ignore cache, models, etc.

│

├── 📁 uploaded_docs/          # Uploaded PDFs (auto-created)

├── 📁 chroma_report_db/       # Chroma vector store (auto-created)

└── 📄 sample.pdf (optional)


## 💬 Modes Explained


| Mode                | Description                      | Accuracy | Cost |
| ------------------- | -------------------------------- | -------- | ---- |
| **OpenAI GPT Mode** | Uses GPT-3.5/4 with user API key | ⭐⭐⭐⭐     | Paid |
| **Local Mode (T5)** | Runs Flan-T5 locally             | ⭐⭐       | Free |


**💡 Your OpenAI API key is never stored. It’s used only during your active session.**


## 🧑‍💻 Author


**👤 Sachin Kumar Gupta**


Data Analyst & AI Developer


🔗 [LinkedIn](linkedin.com/in/sachingupta-ds)
 | [GitHub](https://github.com/Sachin-Kumar-Gupta)


 ## **❤️ Acknowledgements**

**Built using:**

- Streamlit

- ChromaDB

- Sentence Transformers

- Hugging Face Transformers

- OpenAI API
