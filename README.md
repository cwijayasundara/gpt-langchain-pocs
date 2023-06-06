# OpenAI API Invoker Sample App

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/).
2. Clone this repository.

3. Navigate into the project directory:

   ```bash
   $ cd langchain-gpt-invoker
   ```

4. Create a new virtual environment:

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

5. Install the requirements:

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file:

   ```bash
   $ cp .env.example .env
   ```

7. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file.

8. Run the DB chat agent app:

   ```bash
   $ cd banking-db-chat-agent
   $ streamlit run langchain-banking-agent.py
   ```
9. Run the banking-document-chat-agent app:

   ```bash
   $ cd banking-document-chat-agent
   $ streamlit run sec-file-chat-agent.py
   ```
10. Run the banking-document-chat-plain app: This uses a Pinecone client to search the documents

   ```bash
   $ cd banking-document-chat-agent
   $ streamlit run sec-file-chat-plain.py
   ```
   
11. If you find issues with SSL, pls do the following
    ```bash
    $ pip uninstall urllib3
    $ pip install 'urllib3<2.0'
    ```
12. If you want to run the WIKI search agent then do the following
    ```bash
    $ cd wiki-reader-agent
    $ streamlit run wiki-reader-agent.py
    ```
13. If you want to run the SEC file pls use this
    ```bash
    $ cd wiki-reader-agent
    $ streamlit run pinecone-agent.py
14. If you want to run the help desk application, then the following
    ```bash
    $ cd helpdesk-agent
    $ streamlit run helpdesk-automator-langchain.py
    ```