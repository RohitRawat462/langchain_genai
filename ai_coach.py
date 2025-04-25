import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model_name="meta-llama/llama-4-maverick-17b-128e-instruct")

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.messages = []
memory = st.session_state.memory

# Prompt
# coach_prompt = PromptTemplate(
#     input_variables=["combined_input", "chat_history"],
#     template="""
# You are a thoughtful and empathetic AI life coach.

# Conversation history:
# {chat_history}

# The user has selected a category and provided input: {combined_input}

# Ask the user short and simple follow-up 2 questions only to better understand their concerns in this category.
# Once you feel you have enough information, switch to providing a thoughtful final summary and roadmap to help them.

# include this heading: \n🧭 Final Reflection & Roadmap
# """
# )
 # ❗ Ask only 2 follow-up questions maximum — after that, stop asking and provide a helpful final summary and roadmap.

coach_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("""
You are a thoughtful, category-aware AI life coach and mentor. 
You provide personalized guidance in areas like career, relationships, health, personal growth, future planning, interview preparation, and stress management.

Your style is empathetic, supportive, and actionable. 
Ask short, specific follow-up questions to better understand the user's needs. 
Once you feel you understand the user's situation, provide a helpful final summary and roadmap.


Always end with this heading when you're ready: 
🧭 Final Reflection & Roadmap
"""),

    HumanMessagePromptTemplate.from_template("""
Conversation history:
{chat_history}

\n\nUser input: {combined_input}\n\n

Please continue the conversation.
""")
])

# Chain
parser = StrOutputParser()

# Streamlit UI
st.set_page_config(page_title="AI Life Coach", layout="centered")
st.title("💬 AI Life Coach")
st.write("""
Welcome to the AI Life Coach. Select a category, and begin a conversation. 
The AI will ask questions to understand you better and then offer a summary and roadmap.
""")

# Categories
categories = ["relationships", "career", "personal growth", "health", "future planning", "stress management"]
category = st.selectbox("Select a category:", categories)

# Reset Button
if st.button("🔄 Start Over"):
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.rerun()

# Chat interface
if category:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Share your thoughts...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            # Combine category and user input into a single string
            combined_input = f"Category: {category}\nUser input: {user_input}"

            # Initialize the chain
            chain = LLMChain(
                llm=llm,
                prompt=coach_prompt,
                memory=memory,
                verbose=True,
            )

            # Pass the input dictionary with a single key
            #input_data = {"combined_input": combined_input}

            # Invoke the chain
            response = chain.invoke(combined_input)["text"]  # Extract the text from the response

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        if "Final Reflection" in response:
            st.success("✅ Coaching session complete.")