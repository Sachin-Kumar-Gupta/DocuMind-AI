import streamlit as st
import os
from RAG_chatbot import (
    extract_text_from_pdf,
    chunk_text,
    embed_texts,
    create_vector_db,
    ingest_docs,
    answer_with_sources,
    generate_document_summary
)
from chromadb import PersistentClient

# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AI Document Q&A Assistant")
st.markdown("Welcome to **AI Document Assistant** — ask questions about your PDFs.")
st.divider()

# ----------------------------
# Mode selection
# ----------------------------
st.subheader("⚙️ Select Mode")
mode = st.radio("Mode:", ["OpenAI API", "Local Model"], index=1)
user_api_key = None
if mode == "OpenAI API":
    user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    st.info("💡 Responses will use OpenAI GPT.")
else:
    st.warning("⚠️ Using local model may be slower and less fluent.")

# ----------------------------
# Demo PDF
# ----------------------------
st.subheader("🧪 Demo Document")
demo_pdf_path = "uploaded_docs/Report_on_AI.pdf"
if os.path.exists(demo_pdf_path):
    st.info("💡 Demo document is preloaded.")
    st.session_state["current_pdf"] = demo_pdf_path
else:
    st.warning("Demo PDF not found in uploaded_docs folder.")

# ----------------------------
# Upload PDF
# ----------------------------
st.subheader("📤 Upload your document")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    os.makedirs("uploaded_docs", exist_ok=True)
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded `{uploaded_file.name}` successfully!")
    st.session_state["current_pdf"] = file_path

# ----------------------------
# PDF Processing
# ----------------------------
@st.cache_data(show_spinner=False)
def cached_process_pdf(file_path):
    """Extract, chunk, embed, and index PDF."""
    client, collection = create_vector_db(file_path)
    raw_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(raw_text)
    embeddings = embed_texts(chunks)
    ingest_docs(collection, chunks, embeddings, file_path)
    return collection

if "current_pdf" in st.session_state:
    file_path = st.session_state["current_pdf"]
    st.session_state.setdefault("processed_docs", {})
    if file_path not in st.session_state["processed_docs"]:
        with st.spinner("📄 Processing document..."):
            collection = cached_process_pdf(file_path)
            st.session_state["processed_docs"][file_path] = collection
        st.success("✅ Document ready!")
    else:
        collection = st.session_state["processed_docs"][file_path]

# ----------------------------
# Chat Interface
# ----------------------------
if "current_pdf" in st.session_state:
    st.divider()
    st.subheader("💬 Chat with your document")
    st.caption("Try asking questions like: summary, key findings, conclusions.")

    st.session_state.setdefault("chat_history", [])

    # Display chat history
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("Ask your question...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, docs, metas = answer_with_sources(
                        q=user_input,
                        top_k=3,
                        source_file=os.path.basename(st.session_state["current_pdf"]),
                        use_openai=True if user_api_key else False,
                        user_api_key=user_api_key
                    )
                except Exception as e:
                    answer = f"⚠️ Error: {str(e)}"
                    docs, metas = [], []

                st.markdown(answer)

                # Display retrieved chunks
                if docs:
                    with st.expander("🔍 Retrieved Sources"):
                        for i, (doc, meta) in enumerate(zip(docs, metas)):
                            st.markdown(f"**Chunk {meta.get('chunk_id','?')} (Source: {meta.get('source','?')})**")
                            st.write(doc[:300] + ("..." if len(doc) > 300 else ""))

        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

# ----------------------------
# Document Summary
# ----------------------------
if "current_pdf" in st.session_state:
    st.divider()
    st.subheader("📊 Document Summary")
    summary_cache = st.session_state.setdefault("summary_cache", {})

    if st.button("Generate Summary"):
        file_key = st.session_state["current_pdf"]
        if file_key in summary_cache:
            summary = summary_cache[file_key]
        else:
            with st.spinner("📝 Generating summary..."):
                try:
                    summary = generate_document_summary(
                        use_openai=True if user_api_key else False,
                        user_api_key=user_api_key,
                        top_chunks=10
                    )
                except Exception as e:
                    summary = f"⚠️ Error generating summary: {str(e)}"
                summary_cache[file_key] = summary

        st.markdown("### 📄 Document Summary")
        st.write(summary)
        st.success("🎯 Summary generated!")

# ----------------------------
# Reset Chat
# ----------------------------
if st.button("🧹 Reset Chat"):
    st.session_state["chat_history"] = []

st.markdown("---")
st.footer("Built with ❤️ by Sachin Kumar Gupta | Powered by RAG, Sentence Transformers & Streamlit")
