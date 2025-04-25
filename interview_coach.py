import os
import streamlit as st
# from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain

# Load environment variables
# load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model_name="meta-llama/llama-4-maverick-17b-128e-instruct")

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.messages = []
memory = st.session_state.memory

# Streamlit UI
st.set_page_config(page_title="Interview Coach Pro", layout="centered")
st.title("🧠 Interview Coach Pro")
st.write("""
Select a category for interview preparation. The AI will ask questions to assess you, conduct an interview, then give a final summary.
""")

# Categories
categories = ["Software Engineering", "Data Science", "Product Management", "Marketing", "HR", "Finance"]

# Initialize previous category in session state if not set
# if "selected_category" not in st.session_state:
#     st.session_state.selected_category = None
if "just_switched_category" not in st.session_state:
    st.session_state.just_switched_category = False
# Show the dropdown
category = st.selectbox("Select your interview category:", categories)

if "selected_category" not in st.session_state:
    st.session_state.selected_category = category

if category != st.session_state.selected_category:
    st.session_state.selected_category = category
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.session_state.just_switched_category = True
    st.rerun() 


if st.session_state.just_switched_category:
    first_question = f"👋 To begin your **{category}** interview prep, could you briefly describe your experience in this field so I can tailor the interview accordingly?"
    st.session_state.messages.append({"role": "assistant", "content": first_question})
    st.session_state.just_switched_category = False

# Show all messages

# with st.chat_message("assistant"):
#      st.markdown(first_question)



# Reset
if st.button("🔄 Start Over"):
    st.session_state.memory.clear()
    st.session_state.messages = []
    st.rerun()

# Prompt Template
system_template = system_template = """
You are a highly skilled AI interview coach with expertise in {combined_input} interviews.

Your job is to:
- Start by asking which domain or specialization the user wants to focus on within the selected category (e.g., for software: frontend, backend, Android, Java, Python, etc).
- Ask what type of interview they prefer: MCQ, written, or conversational.
- Ask about their experience level in this domain., don't ask Experience Level if already have experience years from the user. If user have answer just number like 5 for exp then you must consider that.
- Ask about their interview goals.
- Then conduct a 3-question mock interview in the selected format.
- Questions should increase in difficulty.
- Be engaging and professional.
- Ask questions in a structured format one at a time so that the user can focus and enjoy answering them.
- Do NOT provide the correct answer after each question, just give next question don't tell that you are right/wrong,
  only reveal all answers at the end.
- Don't ask any repetive questions.
- Please be creative and smart in your questioning. You must conside any answer wriiten by user and you need to understand that.

📝 **Formatting Rules:**
- When presenting multiple-choice questions, always display each option on a separate line using line breaks. Example:

Question text here...

A. Option 1  
B. Option 2  
C. Option 3  
D. Option 4

🧭 Final Reflection & Roadmap:
- Strengths
- Areas for improvement
- Confidence level (1–10)
- Estimated readiness score (1–100)
- Actionable suggestions
- You need to ask a wrap up question like Before we wrap up, is there anything else you'd like help with regarding your interview preparation? Feel free to ask any follow-up questions, clarify doubts, or let me know if you'd like additional practice in a specific area."
I'm here to support your success! 🚀"""

coach_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("""
Conversation history:
{chat_history}

User input: {combined_input}

Please continue the interview.
""")
])

# First question logic - only asked once at the beginning
if "asked_initial_question" not in st.session_state:
    st.session_state.asked_initial_question = False

if category and not st.session_state.asked_initial_question:
    first_question = f"To begin your **{category}** interview prep, could you briefly describe your experience in this area so I can tailor the upcoming questions accordingly?"
    st.session_state.messages.append({"role": "assistant", "content": first_question})
    st.session_state.asked_initial_question = True

# Chat interface
if category:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Share your thoughts to begin your interview prep...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            combined_input = f"Category: {category}, User input: {user_input}"
            chain = LLMChain(
                llm=llm,
                prompt=coach_prompt,
                memory=memory,
                verbose=True
            )

            response = chain.invoke({"combined_input": combined_input})["text"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        if "Final Reflection" in response:
            st.success("✅ Interview summary complete.")
