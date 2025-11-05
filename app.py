import streamlit as st
from RAG_chatbot import chatbot, demo_query
import os
import time

# ----------------------------
# 🎨 Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------
# 🌟 Header Section
# ----------------------------
st.title("🤖 AI Document Q&A Assistant created by Sachin Kumar Gupta")
st.markdown(
    """
    Welcome to **AI Document Assistant** — a smart RAG-powered chatbot that can answer your questions based on your uploaded PDF.  
    Upload your document, choose how you want to run the model, and start asking questions! 🚀
    """
)

st.divider()

# ----------------------------
# 🔐 Mode Selection
# ----------------------------
st.subheader("⚙️ Choose How You Want to Run the Chatbot")

mode = st.radio(
    "Select Mode:",
    ["Use OpenAI API (Recommended)", "Use Normal Mode (Local Model)"],
    index=1
)

user_api_key = None
if mode == "Use OpenAI API (Recommended)":
    st.info(
        "💡 **Note:** Responses in OpenAI mode are generally more accurate, fluent, and context-aware.\n"
        "Enter your OpenAI API key below — it's used only for this session and not stored anywhere."
    )
    user_api_key = st.text_input("🔑 Enter your OpenAI API key:", type="password")
    if not user_api_key:
        st.warning("Please enter your OpenAI API key to use this mode.")
else:
    st.warning(
        "⚠️ You are using Normal Mode (Local Model). "
        "Responses may not always be perfectly accurate or detailed. "
        "For professional use, please switch to OpenAI mode."
    )

st.divider()

@st.cache_resource(show_spinner=False)
def process_document(file_path):
    chatbot(file_path)
    return True

# ----------------------------
# 📄 Document Upload
# ----------------------------
st.subheader("📤 Upload your document")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf","docx","doc"])

if uploaded_file:
    upload_dir = "/tmp/uploaded_docs"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    os.makedirs(upload_dir, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ Uploaded `{uploaded_file.name}` successfully!")
    st.info("🔍 Processing document... Please wait ⏳")

    with st.spinner("🔎 Reading, chunking, and embedding your document..."):
        process_document(file_path)
        time.sleep(2)

    st.success("🎯 Document processed and ready for questions!")

    st.divider()

    # ----------------------------
    # 💬 Chat Interface
    # ----------------------------
    st.subheader("💬 Ask Questions About Your Document")
    question = st.text_input("Type your question here:")

    if st.button("Ask"):
        if question.strip():
            with st.spinner("🤔 Thinking..."):
                answer = demo_query(
                    question,
                    user_api_key=user_api_key if user_api_key else None,
                    top_k=3,
                    use_openai=True if user_api_key else False
                )

            st.markdown("### 🧠 **Answer:**")
            st.write(answer)
        else:
            st.warning("Please enter a question before clicking *Ask*.")

else:
    st.info("📄 Please upload a PDF file to begin.")

# ----------------------------
# 🧾 Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by Sachin Kumar Gupta | Powered by RAG, Sentence Transformers & Streamlit")
