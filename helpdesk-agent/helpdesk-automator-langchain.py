import os
import openai
import warnings
import pandas as pd
import streamlit as st

from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

warnings.filterwarnings('ignore')

llm = ChatOpenAI(temperature=0.8)

# load the dataset from huggingface
dataset = load_dataset("banking77")

# Sort the dataset by the length of the customer texts
sorted_data = sorted(dataset['train'], key=lambda x: len(x['text']), reverse=True)

longest_ten_texts = [entry["text"] for entry in sorted_data[:10]]

# print the values in longest_ten_texts one at a time
for text in longest_ten_texts:
    print(text)
    print()

df = pd.DataFrame(longest_ten_texts, columns=['Customer Enquiry'])
df.rename(columns={'text': 'Customer Enquiry'})

st.table(df)

selected_indices = st.multiselect('Select Rows:', df.index)
selected_rows = df.loc[selected_indices]
st.write('### Selected Rows', selected_rows)
selected_rows = df.loc[selected_indices]


# SequentialChain
english_translator_prompt = ChatPromptTemplate.from_template(
    "Translate the following enquiry to english:{Review}")

# chain 1: input= Review and output= English_Review
english_translator_chain = LLMChain(llm=llm, prompt=english_translator_prompt, output_key="English_Review")

# summary chain
summary_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following enquiry in no longer than 100 words?: {English_Review}")

# chain 2: input= English_Review and output= summary
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# sentiment chain
sentiment_prompt = ChatPromptTemplate.from_template("Identify the sentiment of the the following enquiry in single "
                                                    "word, positive, negative or neutral: {summary}")

sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt, output_key="sentiment")

# Intent chain
intent_prompt = ChatPromptTemplate.from_template("Identify the intent of the the following enquiry in single sentence"
                                                 "\n\n{summary}"
                                                 )
intent_chain = LLMChain(llm=llm, prompt=intent_prompt, output_key="intent")

# Identity the original language the enquiry was written in
language_prompt = ChatPromptTemplate.from_template("What language is the following enquiry:\n\n{Review}")

# input= Review and output= language
language_chain = LLMChain(llm=llm, prompt=language_prompt, output_key="language")

# prompt template 4: follow-up message
response_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response in very professionally to the following "
    "summary and sentiment in the specified language:"
    "\n\nSummary: {summary}\n\nsentiment: {sentiment}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
response_chain = LLMChain(llm=llm, prompt=response_prompt, output_key="followup_message")

# overall_chain: input= Review
# and output= English_Review,summary, follow up_message
overall_chain = SequentialChain(
    chains=[english_translator_chain, summary_chain, sentiment_chain, intent_chain, language_chain, response_chain],
    input_variables=["Review"],
    output_variables=["English_Review", "summary", "sentiment", "intent", "language", "followup_message"],
    verbose=True
)

# loop through the selected rows and print the text
for index, row in selected_rows.iterrows():
    customer_review_text = row['Customer Enquiry']
    response = overall_chain(customer_review_text)
    st.write(response)