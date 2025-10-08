import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import os

load_dotenv()  # Make sure your GROQ_API_KEY is in .env

# --- Streamlit setup ---
st.set_page_config(page_title="Groq LLaMA2 Chatbot", page_icon="⚡")
st.title("⚡ Groq-Powered LLaMA2 Chatbot")

# --- LangChain LLM Setup with Groq ---
llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b",  # or mistral-7b / mixtral-8x7b / gemma-7b
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("GROQ_API_KEY"),  # preferred
    temperature=0.5,
)

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(content="You are a helpful assistant.")
    ]

# --- User Input ---
user_input = st.text_input("Ask something:", key="input")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get response from Groq
    with st.spinner("Thinking with Groq..."):
        response = llm(st.session_state.chat_history)

    # Add assistant message to history
    st.session_state.chat_history.append(response)

# --- Display chat ---
for msg in reversed(st.session_state.chat_history[1:]):  # Skip system prompt
    role = "You" if msg.type == "human" else "LLaMA2"
    st.markdown(f"**{role}:** {msg.content}")
