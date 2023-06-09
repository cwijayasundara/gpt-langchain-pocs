import os
import pinecone
import warnings
import openai
import streamlit as st

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner
)

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # find next to api key in console
PINECONE_ENV = os.getenv('PINECONE_ENV')  # find next to api key in console
index_name = 'semantic-search-openai'
warnings.filterwarnings('ignore')
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

# create the LLM
llm = ChatOpenAI(temperature=0.0)

# embedding model
embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=openai.api_key
)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
# connect to index assuming its already created
index = pinecone.Index(index_name)
print('Pinecone index status is', index.describe_index_stats())

text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
)

# create an agent
tools = load_tools(['wikipedia', 'ddg-search','llm-math'], llm=llm)
memory = ConversationBufferMemory(memory_key='chat_history')

# Use the Vector DB as a tool
name = """Alphabet quarterly earning reports database"""

description = ("\n"
               "Useful for when you need to answer questions about the earnings of Google and Alphabet in 2021, "
               "2022 and 2023. Input may be a partial or fully formed question.\n")
vector_db_search_tool = Tool(
    name=name,
    func=qa.run,
    description=description,
)

tools.append(vector_db_search_tool)

planner = load_chat_planner(llm)
executor = load_agent_executor(llm, tools, verbose=True)

agent = PlanAndExecute(
    planner=planner,
    executor=executor,
    verbose=True,
    reduce_k_below_max_tokens=True
)

st.title("Alphabet Investment Advisor Bot!")
st.text_input("Please Enter Your Query On Investing in Alphabet! ", key="query")
result = agent.run(st.session_state.query)
st.write(result)




