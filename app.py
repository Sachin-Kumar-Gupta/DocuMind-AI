import streamlit as st
from RAG_chatbot import chatbot, demo_query, answer_with_sources, generate_document_summary
import os
import time

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

st.divider()

st.subheader("🧪 Try a Demo Document")

demo_pdf_path = "docs/Reports_on_ai.pdf"

if os.path.exists(demo_pdf_path):

    if st.button("📄 Load Demo Document"):

        if not st.session_state.get("demo_loaded"):
            with st.spinner("Processing demo document..."):
                chatbot(demo_pdf_path)
    
            st.session_state["demo_loaded"] = True
    
            st.success("Demo document ready for questions!")
        else:
            st.info("Demo document already loaded.")

else:
    st.warning("Demo PDF not found. Add one to demo_docs folder.")



# ----------------------------
# 📄 Document Upload
# ----------------------------
st.subheader("📤 Upload your document")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file or st.session_state.get("demo_loaded") :
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    os.makedirs("uploaded_docs", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded `{uploaded_file.name}` successfully!")
    st.info("🔍 Processing document... Please wait ⏳")

    progress = st.progress(0)

    with st.spinner("Extracting document text..."):
        progress.progress(25)
        time.sleep(0.5)
    
    with st.spinner("Chunking document..."):
        progress.progress(50)
        time.sleep(0.5)
    
    with st.spinner("Creating embeddings & index..."):
        chatbot(file_path)
        progress.progress(100)

    st.success("🎯 Document processed and ready for chat!")

    st.divider()

    # ----------------------------
    # 💬 Chat Interface
    # ----------------------------
    st.subheader("💬 Chat with your document")
    st.caption("💡 Try asking:")
    st.markdown("""
    - What is the main topic of this document?
    - Summarize the key findings
    - What conclusions are mentioned?
    """)

    # Initialize session history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display all previous messages
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Input for new user question
    user_input = st.chat_input("Ask your question about the document...")

    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})

        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer, docs, metas = answer_with_sources(
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
        # Store bot reply in session
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

else:
    st.info("📄 Please upload a PDF file to begin.")

st.subheader("📊 Document Insights")

if st.button("Generate Document Summary"):
    with st.spinner("Analyzing document..."):
        summary = generate_document_summary(
            use_openai=True if user_api_key else False,
            user_api_key=user_api_key
        )
    st.markdown("### 📄 Document Summary")
    st.write(summary)
    st.success("🎯 Document processed and ready for chat!")

    file_size = os.path.getsize(file_path) / 1024
    st.caption(f"Document size: {file_size:.1f} KB")

if st.button("🧹 Reset Chat"):
    st.session_state["chat_history"] = []


# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by Sachin Kumar Gupta | Powered by RAG, Sentence Transformers & Streamlit")



