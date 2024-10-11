## LLaMA 2 based conversational AI chatbot

- This repo builds LLaMA 2 based chatbot for querying local documents. 
- Tools used: Ollama, Langchain, Streamlit


### Step 1: Build Environment
1. Download Ollama : [Link](https://ollama.com/)
2. Pull LLaMA2 model: ollama pull llama2
3. Create python virtual environment: ENV_NAME=venv_llama2 && python -m venv $ENV_NAME && source $ENV_NAME/bin/activate
4. Install dependencies: pip install -r requirements.txt

### Step 2: 
1. Place your documents in docs/ foler
2. Run: streamlit run query_local_docs.py


### Future Steps
- Create vector database for local files
- Fine tune LLaMA2 on these files locally
- Integrate cloud storage (google drive) to search through files and folders