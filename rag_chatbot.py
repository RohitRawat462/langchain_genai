import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import tempfile
import shutil
import chromadb
import time
from langchain_openai import OpenAIEmbeddings


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Set up Streamlit page
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="wide")
st.title("📄 Ask Questions")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = None
if "persist_directory" not in st.session_state:
    st.session_state.persist_directory = "./chroma_db"

# Function to safely reset Chroma database
def reset_chroma_db():
    max_retries = 3
    retry_delay = 1  # seconds
    for attempt in range(max_retries):
        try:
            # Close existing Chroma client
            if st.session_state.chroma_client is not None:
                st.session_state.chroma_client = None
            # Delete Chroma directory
            if os.path.exists(st.session_state.persist_directory):
                shutil.rmtree(st.session_state.persist_directory)
                st.write(f"Cleared {st.session_state.persist_directory}")
                return True
        except PermissionError as e:
            st.warning(f"Attempt {attempt + 1}/{max_retries}: Could not clear {st.session_state.persist_directory}: {e}. Retrying after {retry_delay}s...")
            time.sleep(retry_delay)
        except Exception as e:
            st.error(f"Error resetting Chroma database: {e}")
            return False
    # Fallback: Use a new directory
    st.warning(f"Failed to clear {st.session_state.persist_directory} after {max_retries} attempts. Using a new directory.")
    st.session_state.persist_directory = f"./chroma_db_{int(time.time())}"
    st.write(f"Switched to new directory: {st.session_state.persist_directory}")
    return True

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf", 'txt'])

if uploaded_file:
    # Reset for new file
    if st.session_state.last_uploaded_file != uploaded_file.name:
        reset_chroma_db()
        st.session_state.qa_chain = None
        st.session_state.memory.clear()
        st.session_state.chat_history = []
        st.session_state.last_uploaded_file = uploaded_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF Uploaded Successfully!")

    # Initialize chain
    if st.session_state.qa_chain is None:
        with st.spinner("🔍 Processing your PDF..."):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            # Create new Chroma client
            st.session_state.chroma_client = chromadb.PersistentClient(path=st.session_state.persist_directory)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                #embedding=OllamaEmbeddings(model="nomic-embed-text"),
                embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
                persist_directory=st.session_state.persist_directory,
                #client=st.session_state.chroma_client
            )

            st.session_state.vectorstore = vectorstore

            retriever = vectorstore.as_retriever()

            llm = ChatGroq(
                api_key=groq_api_key,
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
            )

            # Custom prompt with system prompt
            prompt_template = """
            System: You are a knowledgeable assistant specializing in answering questions based on PDF documents. Use the provided conversation history and retrieved document context to give accurate, concise, and context-aware answers. If the question relates to previous interactions, reference the history to maintain continuity.

            Conversation History:
            {chat_history}
            Question: {question}
            Context: {context}
            Answer:
            """
            prompt = PromptTemplate(
                input_variables=["chat_history", "question", "context"],
                template=prompt_template
            )

            # Initialize chain
            st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True,
                output_key="answer"
            )

    # Query input
    st.subheader("💬 Ask a question from the PDF")
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input("Type your question:")
        submitted = st.form_submit_button("🚀 Send")

    if submitted and query:
        with st.spinner("🤖 Thinking..."):
            try:

                # get similiarity score and page content 
                similarity_score = st.session_state.vectorstore.similarity_search_with_score(query, k=5)
                st.write(similarity_score)
                for doc, score in similarity_score:
                  st.write(f"Score: {score}")
                  st.write(f"Page Content: {doc.page_content}")

                # Run query
                result = st.session_state.qa_chain.invoke({"question": query})

                # Update chat history
                st.session_state.chat_history.append({"question": query, "answer": result["answer"]})

                # Display answer
                st.markdown(f"### 📌 Answer:\n{result['answer']}")

                # Show sources
                with st.expander("📄 Source(s)"):
                    for doc in result["source_documents"]:
                        st.write(doc.metadata['source'])

                # Debug memory
                st.write("**Memory Contents:**")
                st.write(st.session_state.memory.load_memory_variables({}))

            except Exception as e:
                st.error(f"Error: {e}")

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for i, chat in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i+1}:** {chat['question']}")
            st.write(f"**A{i+1}:** {chat['answer']}")
            st.markdown("---")

# Clean up
if uploaded_file and os.path.exists(pdf_path):
    os.unlink(pdf_path)

# Reset button
if st.button("🗑️ Reset Conversation"):
    reset_chroma_db()
    st.session_state.memory.clear()
    st.session_state.qa_chain = None
    st.session_state.chat_history = []
    st.session_state.last_uploaded_file = None
    st.session_state.chroma_client = None
    st.session_state.persist_directory = "./chroma_db"  # Reset to default
    st.success("Conversation reset!")