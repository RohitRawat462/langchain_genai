import os
import streamlit as st
from PIL import Image
import easyocr
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variable
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# Streamlit page setup
st.set_page_config(page_title="🖼️ Image Q&A App", layout="centered")
st.title("🖼️ Extract Info from Images using OCR + LLM")
st.write("Upload an image, we'll extract the text and let you ask questions!")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process image with OCR
    with st.spinner("🔍 Extracting text from image..."):
        reader = easyocr.Reader(['en'], gpu=False)
        image = Image.open(uploaded_file)
        result = reader.readtext(image, detail=0)
        extracted_text = "\n".join(result)

        st.subheader("📝 Extracted Text:")
        st.code(extracted_text)

        # Create LangChain document
        docs = [Document(page_content=extracted_text, metadata={"source": uploaded_file.name})]

        # Split and embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(chunks, embedding=OllamaEmbeddings(model="nomic-embed-text"))

        # Retrieval QA setup
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        # Ask question
        query = st.text_input("❓ Ask a question about the image content:")
        if st.button("🔍 Submit") and query:
            with st.spinner("🤖 Thinking..."):
                response = qa_chain({"query": query})
                st.markdown("### ✅ Answer:")
                st.write(response["result"])

                st.markdown("### 📄 Source(s):")
                st.write([doc.metadata["source"] for doc in response["source_documents"]])
