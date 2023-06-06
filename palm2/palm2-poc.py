import vertexai

from langchain import PromptTemplate, LLMChain
from google.cloud import aiplatform

PROJECT_ID = "ibm-keras"
LOCATION = "us-west1"  # e.g. us-central1

vertexai.init(project=PROJECT_ID, location=LOCATION)

template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = VertexAI()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)