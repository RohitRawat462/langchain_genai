import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import SerpAPIWrapper, WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain.memory import ConversationBufferMemory

# For file handling
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from PIL import Image
import pytesseract

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-70b-8192")

# Define tools
search_tool = Tool(
    name="Real-Time Web Search",
    func=SerpAPIWrapper().run,
    description="Search the web for current information using SerpAPI"
)

wiki_tool = Tool(
    name="Wikipedia",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
    description="Fetch information from Wikipedia for factual knowledge."
)

python_tool = PythonREPLTool()

tools = [search_tool, wiki_tool, python_tool]

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
memory = st.session_state.memory

# Initialize Agent
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# Streamlit UI
st.set_page_config(page_title="LangChain AI Assistant", layout="centered")
st.title("🧠 LangChain Smart AI Assistant")
st.write("This assistant can search the web, use Wikipedia, run Python code, and answer questions from uploaded files.")

# Upload file
uploaded_file = st.file_uploader("📁 Upload a file (PDF, TXT, or Image)", type=["pdf", "txt", "png", "jpg", "jpeg"])

# Handle File Input
if uploaded_file:
    file_type = uploaded_file.type
    if "pdf" in file_type:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"))
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
        st.session_state.rag_chain = rag_chain
        st.success("✅ PDF loaded and ready for questions!")

    elif "text" in file_type:
        with open("temp.txt", "wb") as f:
            f.write(uploaded_file.read())
        loader = TextLoader("temp.txt")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"))
        rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)
        st.session_state.rag_chain = rag_chain
        st.success("✅ Text file loaded and ready!")

    elif "image" in file_type:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        st.write("📝 Extracted Text from Image:")
        st.code(text)
        st.session_state.extracted_text = text

# User input
query = st.text_input("Ask your question:", key="input")
submit_button = st.button("🔍 Submit")

if submit_button and query:
    with st.spinner("Thinking..."):
        try:
            # If PDF/Text file was uploaded, use RAG
            if "rag_chain" in st.session_state:
                result = st.session_state.rag_chain.invoke({"query": query})
                st.markdown("### 📘 RAG Answer:")
                st.write(result["result"])
                st.markdown("**📄 Sources:**")
                for doc in result["source_documents"]:
                    st.write("-", doc.metadata.get("source", "Unknown source"))

            # If image text was extracted
            elif "extracted_text" in st.session_state:
                prompt = f"{st.session_state.extracted_text}\n\nQuestion: {query}"
                result = llm.invoke(prompt)
                st.markdown("### 🖼️ Answer from Image:")
                st.write(result.content)

            # Otherwise use normal agent tools
            else:
                result = agent_executor.invoke({"input": query})
                st.markdown("### 🤖 Agent Answer:")
                st.write(result["output"])

        except Exception as e:
            st.error(f"Error: {e}")
