import os
import openai
import warnings
import pinecone
import streamlit as st
import warnings

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

warnings.filterwarnings('ignore')

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import load_tools, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.vectorstores import Pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent

llm = ChatOpenAI(temperature=0.0)

EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'
warnings.filterwarnings('ignore')

embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=openai.api_key
)

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # find next to api key in console
PINECONE_ENV = os.getenv('PINECONE_ENV')  # find next to api key in console

index_name = 'semantic-search-openai'
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
# connect to index
index = pinecone.Index(index_name)
print('Pinecone index status is', index.describe_index_stats())

text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering queries about '
            'the SEC files'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

print(agent('Whats the total revenue of Tesla for 2022?'))

st.title('Pinecone Agent')
st.text_input("Please Enter Your SEC File Query! ", key="query")
try:
    result = qa(st.session_state.query)
    print(result)
    st.write(result)
except:
    st.write("")

# Need to write the evaluations for the agent
