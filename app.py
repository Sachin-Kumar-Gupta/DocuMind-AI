import streamlit as st
from RAG_chatbot import chatbot, demo_query, answer_with_sources, generate_document_summary
from chromadb import PersistentClient
import os
import time

# ---------------------------
# 🔹 ChromaDB setup
# ---------------------------
DEMO_CHROMA_DIR = "./chroma_demo_db"
DEMO_COLLECTION_NAME = "demo_collection"
demo_pdf_path = "uploaded_docs/Report_on_AI.pdf"

client = PersistentClient(path=DEMO_CHROMA_DIR)
collection = client.get_or_create_collection(DEMO_COLLECTION_NAME)

# ---------------------------
# 🔹 Process PDF (with caching)
# ---------------------------
def process_pdf(file_path, show_progress=True):
    st.session_state["current_pdf"] = file_path
    if "processed_docs" not in st.session_state:
        st.session_state["processed_docs"] = {}

    if file_path not in st.session_state["processed_docs"]:
        # Only show progress for user-uploaded PDFs
        if show_progress:
            progress = st.progress(0)
            status = st.empty()
            status.text("📄 Extracting text from document...")
            progress.progress(25)
            time.sleep(0.2)
            status.text("✂️ Chunking document...")
            progress.progress(50)
            time.sleep(0.2)
            status.text("🧠 Generating embeddings...")
            progress.progress(75)
            time.sleep(0.2)
            status.text("📚 Building vector index...")

        # Call chatbot to index PDF (will skip if DB already exists)
        chatbot(file_path)

        if show_progress:
            progress.progress(100)
            status.text("✅ Document ready!")

        st.session_state["processed_docs"][file_path] = True
    else:
        st.info("✅ Document already processed")

# ----------------------------
# 🔹 Cached answers per session
# ----------------------------
def get_answer_cached(question, top_k=3, use_openai=False, user_api_key=None):
    if "answer_cache" not in st.session_state:
        st.session_state["answer_cache"] = {}
    cache_key = f"{question}_{top_k}_{use_openai}"
    if cache_key in st.session_state["answer_cache"]:
        return st.session_state["answer_cache"][cache_key]
    ans = answer_with_sources(
        question,
        top_k=top_k,
        use_openai=use_openai,
        user_api_key=user_api_key
    )
    st.session_state["answer_cache"][cache_key] = ans
    return ans

# ----------------------------
# 🎨 Streamlit Page Config
# ----------------------------
st.set_page_config(page_title="AI Document Q&A Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Document Q&A Assistant")
st.markdown("""
Welcome to **AI Document Assistant** — a smart RAG-powered chatbot that can answer your questions based on your uploaded PDF.  
Upload your document, choose your mode, and chat naturally with your document! 🚀
""")
st.info("""
📌 **How to use this app**

1️⃣ Upload a PDF document  
2️⃣ Wait for document indexing  
3️⃣ Ask questions about the content  
4️⃣ View retrieved sources for transparency  

Tip: Try asking **summary, key findings, or conclusions**
""")
st.divider()

# ----------------------------
# ⚙️ Mode Selection
# ----------------------------
st.subheader("⚙️ Choose How You Want to Run the Chatbot")
mode = st.radio("Select Mode:", ["Use OpenAI API (Recommended)", "Use Normal Mode (Local Model)"], index=1)
user_api_key = None

if mode == "Use OpenAI API (Recommended)":
    st.info("💡 OpenAI mode gives more fluent, accurate responses. Enter your API key (not stored).")
    user_api_key = st.text_input("🔑 Enter your OpenAI API key:", type="password")
else:
    st.warning("⚠️ Local mode may not always be perfectly accurate — use OpenAI mode for professional work.")

# ----------------------------
# 🧪 Demo PDF
# ----------------------------
st.subheader("🧪 Demo Document")
if os.path.exists(demo_pdf_path):
    if len(collection.get()) > 0:
        st.info("💡 Demo document is preloaded. Chat instantly!")
        st.session_state["demo_loaded"] = True
        st.session_state["current_pdf"] = demo_pdf_path
    else:
        st.info("💡 Preprocessing demo document for first time...")
        process_pdf(demo_pdf_path, show_progress=False)
        st.session_state["demo_loaded"] = True
        st.session_state["current_pdf"] = demo_pdf_path
else:
    st.warning("Demo PDF not found. Add one to uploaded_docs folder.")

# ----------------------------
# 📄 Document Upload
# ----------------------------
st.subheader("📤 Upload your document")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    os.makedirs("uploaded_docs", exist_ok=True)
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded `{uploaded_file.name}` successfully!")
    process_pdf(file_path, show_progress=True)

# ----------------------------
# 💬 Chat Interface
# ----------------------------
if st.session_state.get("current_pdf") or st.session_state.get("demo_loaded"):
    st.divider()
    st.subheader("💬 Chat with your document")
    st.caption("💡 Try asking:")
    st.markdown("""
    - What is the main topic of this document?
    - Summarize the key findings
    - What conclusions are mentioned?
    """)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display previous messages
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Chat input
    user_input = st.chat_input("Ask your question about the document...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer, docs, metas = get_answer_cached(
                    user_input,
                    user_api_key=user_api_key if user_api_key else None,
                    top_k=3,
                    use_openai=True if user_api_key else False
                )
                st.markdown(answer)
                # Show retrieved context
                with st.expander("🔍 Retrieved Sources"):
                    for i, (doc, meta) in enumerate(zip(docs, metas)):
                        st.markdown(f"**Source Chunk {meta['chunk_id']}**")
                        st.write(doc[:500] + "...")
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        # ----------------------------
        # 🧹 Reset Chat (Below everything)
        # ----------------------------
        st.divider()
        if st.button("🧹 Reset Chat"):
            st.session_state["chat_history"] = []
            st.success("Chat history cleared!")

# ----------------------------
# 📊 Document Summary
# ----------------------------
if st.session_state.get("current_pdf") or st.session_state.get("demo_loaded"):
    st.divider()
    st.subheader("📊 Document Insights / Summary")
    if st.button("Generate Document Summary"):
        with st.spinner("Analyzing document..."):
            summary = generate_document_summary(
                use_openai=True if user_api_key else False,
                user_api_key=user_api_key
            )
        st.markdown("### 📄 Document Summary")
        st.write(summary)
        st.success("🎯 Document summary generated!")

# ----------------------------
# 🧹 Reset Chat (Below everything)
# ----------------------------
st.divider()
if st.button("🧹 Reset Chat"):
    st.session_state["chat_history"] = []
    st.success("Chat history cleared!")

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by Sachin Kumar Gupta | Powered by RAG, Sentence Transformers & Streamlit")
