import pinecone
import openai

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# Use the Tesla 2022 10-K report
loader = OnlinePDFLoader("https://ir.tesla.com/_flysystem/s3/sec/000095017023001409/tsla-20221231-gen.pdf")
data = loader.load()

print(f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} charactors in the document')

text_spiltter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_spiltter.split_documents(data)
print(f'Now you have {len(texts)} documents')

print(f'The first chunk of the 10-K file is *** {texts[0]}')

#  create embeddings from the documents
openai.api_key = ""
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
# pinecone index name
index_name = 'semantic-search-openai'

pinecone.init(
    api_key="",
    environment=""  # find next to api key in console
)
# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)
# connect to index
index = pinecone.Index(index_name)

# add documents to index: should be a one time activity!!
docsearch = Pinecone.from_texts([t.page_content for t in texts],embeddings, index_name=index_name )

print('Pinecode index status is', index.describe_index_stats())