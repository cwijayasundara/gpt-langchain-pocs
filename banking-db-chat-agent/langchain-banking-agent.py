import os
import db_setup
import streamlit as st
import openai
import warnings

from dotenv import load_dotenv, find_dotenv
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.sql_database import SQLDatabase
from langchain.chains import SQLDatabaseChain
from langchain.agents import AgentType

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

warnings.filterwarnings('ignore')

llm = OpenAI(temperature=0.0)


def count_tokens(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')
    return result


# set up the database
db_setup.set_up_db()

db = SQLDatabase(db_setup.engine)
sql_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

sql_tool = Tool(
    name='Banking DB',
    func=sql_chain.run,
    description="Useful for when you need to answer questions about retail banking customers " \
                "and their accounts."
)

tools = load_tools(
    ["llm-math"],
    llm=llm
)

tools.append(sql_tool)

memory = ConversationBufferMemory(memory_key="chat_history")

conversational_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent='conversational-react-description',
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=3,
    memory=memory
)

st.title("Chat DB Agent")
st.text_input("Please Enter Your Query! ", key="query")
try:
    result = conversational_agent(st.session_state.query)
    print(result)
    st.write(result)
except:
    print("")

