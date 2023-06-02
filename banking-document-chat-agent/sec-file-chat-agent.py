import streamlit as st
import pinecone
import openai
import os
import warnings

from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']
warnings.filterwarnings('ignore')
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

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

llm = OpenAI(temperature=0.0)

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Create a list of tools for the LangChain agent
tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'Use this tool when answering general knowledge queries to get '
            'more information about the SEC file'
        )
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent='conversational-react-description',
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=3,
    memory=conversational_memory
)

st.title("SEC Files Chat Agent")

st.text_input("Please Enter Your Query! ", key="query")

result = agent(st.session_state.query)

st.write(result)
