import os
import openai
import warnings
import streamlit as st
import warnings

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

warnings.filterwarnings('ignore')

from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType

llm = ChatOpenAI(temperature=0.0)

tools = load_tools(["ddg-search", "wikipedia", "python_repl"])

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False)

st.title("WIKI Chat Bot")
st.text_input("Please Enter Your Query to search WIKI! ", key="query")
try:
    result = agent(st.session_state.query)
    print(result)
    st.write(result)
except:
    print("exception on external access")
