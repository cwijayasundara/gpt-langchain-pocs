import streamlit as st
import pinecone
import openai
import os
import warnings

from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

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

MODEL_NAME = 'gpt-3.5-turbo'
# completion llm
llm = ChatOpenAI(
    openai_api_key=openai.api_key,
    model_name=MODEL_NAME,
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

st.title("SEC Files Chat Bot")
st.text_input("Please Enter Your SEC File Query! ", key="query")
try:
    result = qa(st.session_state.query)
    print(result)
    st.write(result)
except:
    st.write("")
