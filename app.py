import streamlit as st
from RAG_chatbot import chatbot, demo_query
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

# ----------------------------
# 📄 Document Upload
# ----------------------------
st.subheader("📤 Upload your document")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    os.makedirs("uploaded_docs", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded `{uploaded_file.name}` successfully!")
    st.info("🔍 Processing document... Please wait ⏳")

    with st.spinner("Extracting and indexing content..."):
        chatbot(file_path)
        time.sleep(2)

    st.success("🎯 Document processed and ready for chat!")

    st.divider()

    # ----------------------------
    # 💬 Chat Interface
    # ----------------------------
    st.subheader("💬 Chat with your document")

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
                answer = demo_query(
                    user_input,
                    user_api_key=user_api_key if user_api_key else None,
                    top_k=3,
                    use_openai=True if user_api_key else False
                )
                st.markdown(answer)
        # Store bot reply in session
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

else:
    st.info("📄 Please upload a PDF file to begin.")

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by Sachin Kumar Gupta | Powered by RAG, Sentence Transformers & Streamlit")


